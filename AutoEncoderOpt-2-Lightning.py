import os
import numpy as np
import torch
from torch.nn import Linear, MSELoss, BCELoss, ReLU, Dropout, Sequential
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, MessagePassing, GATv2Conv, LayerNorm
from sklearn.model_selection import train_test_split
from scipy import sparse
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random
import trimesh
import open3d as o3d
from itertools import combinations
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_fabric.utilities.rank_zero import rank_zero_only  # For decorator option

import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'png'

# === Configuration ===
DATA_PATH = "Graphs"
CHECKPOINT_DIR_BASE = "lightning_checkpoints"
LOG_DIR_BASE = "lightning_logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Dataset Loader ===
class CarGraphDataset(Dataset):
    def __init__(self, root_dir, indices=None, file_limit=None):
        self.root_dir = Path(root_dir)
        all_files_unsorted = [f for f in self.root_dir.rglob("*.npz") if "_adj" not in f.name]
        all_files_unsorted.sort(
            key=lambda f: int(f.stem.split('_')[-1]) if f.stem.split('_')[-1].isdigit() else float('inf'))
        if indices is not None:
            self.files = [all_files_unsorted[i] for i in indices if i < len(all_files_unsorted)]
        else:
            self.files = all_files_unsorted
        if file_limit is not None: self.files = self.files[:file_limit]
        super().__init__()

    def len(self):
        return len(self.files)

    def get(self, idx):
        feat_file = self.files[idx]
        adj_file = feat_file.with_name(feat_file.name.replace(".npz", "_adj.npz"))
        if not feat_file.exists() or not adj_file.exists():
            # This print might come from multiple dataloader workers if num_workers > 0
            # For infrequent warnings, it's often left as is.
            print(f"Warning: File not found for index {idx}, expected {feat_file}")
            return Data(x=torch.empty(0, 10), edge_index=torch.empty(2, 0, dtype=torch.long),
                        center_point=torch.zeros(3), scale=torch.tensor(1.0), cd=torch.tensor(0.0), filepath="dummy")
        npz_data = np.load(feat_file)
        x = torch.tensor(npz_data["x"], dtype=torch.float32)
        center_point = torch.tensor(npz_data["center_point"], dtype=torch.float32)
        scale = torch.tensor(npz_data["scale"], dtype=torch.float32)
        cd = torch.tensor(npz_data["cd"], dtype=torch.float32)
        a = sparse.load_npz(adj_file).tocoo()
        edge_index = torch.tensor(np.vstack((a.row, a.col)), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, center_point=center_point, cd_value=cd, scale=scale,
                    filepath=str(feat_file))


# === Models ===
class CdLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.0):
        super().__init__()
        layers = [Linear(in_features, 4 * in_features), ReLU()]
        if dropout_rate > 0: layers.append(Dropout(dropout_rate))
        layers.extend([Linear(4 * in_features, 2 * in_features), ReLU()])
        if dropout_rate > 0: layers.append(Dropout(dropout_rate))
        layers.extend([Linear(2 * in_features, in_features), ReLU()])
        if dropout_rate > 0: layers.append(Dropout(dropout_rate))
        layers.append(Linear(in_features, out_features))
        # No final ReLU here to strictly match original notebook model definition for baseline
        # User can add it back: if add_final_relu: layers.append(ReLU())
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GraphAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.res_proj = Linear(in_channels, latent_dim)
        self.gnn_block1_conv = GATv2Conv(in_channels, hidden_channels)
        self.gnn_block1_act = ReLU()
        self.gnn_block1_norm = LayerNorm(hidden_channels)
        self.gnn_block2_conv = GATv2Conv(hidden_channels, 2 * hidden_channels // 4, heads=4, concat=True, residual=True)
        self.gnn_block2_act = ReLU()
        self.gnn_block2_norm = LayerNorm(2 * hidden_channels)
        self.gnn_block3_conv = GATv2Conv(2 * hidden_channels, 4 * hidden_channels // 4, heads=4, concat=True,
                                         residual=True)
        self.gnn_block3_act = ReLU()
        self.gnn_block3_norm = LayerNorm(4 * hidden_channels)
        self.encoder_lin = Sequential(Linear(4 * hidden_channels, latent_dim), ReLU(), Dropout(p=0.2))
        self.decoder_lin = Sequential(
            Linear(latent_dim, 2 * hidden_channels), ReLU(),
            Linear(2 * hidden_channels, hidden_channels), ReLU(),
            Linear(hidden_channels, in_channels))

    def encode(self, x, edge_index, batch=None):
        x_in = x
        x = self.gnn_block1_conv(x, edge_index);
        x = self.gnn_block1_act(x);
        x = self.gnn_block1_norm(x)
        x = self.gnn_block2_conv(x, edge_index);
        x = self.gnn_block2_act(x);
        x = self.gnn_block2_norm(x)
        x = self.gnn_block3_conv(x, edge_index);
        x = self.gnn_block3_act(x);
        x = self.gnn_block3_norm(x)
        z_nodes = self.encoder_lin(x)
        z_nodes = z_nodes + self.res_proj(x_in)
        if batch is None: batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        z_graph = global_mean_pool(z_nodes, batch)
        return z_nodes, z_graph

    def decode(self, z_nodes): return self.decoder_lin(z_nodes)

    def forward(self, x, edge_index, batch=None):
        z_nodes, z_graph = self.encode(x, edge_index, batch)
        x_hat = self.decode(z_nodes)
        return z_nodes, x_hat, z_graph


# === PyTorch Lightning DataModule ===
class CarDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, num_workers=4, random_state=42, file_limit=None):
        super().__init__()
        self.save_hyperparameters()
        self.root_dir, self.num_workers, self.random_state, self.file_limit = root_dir, num_workers, random_state, file_limit
        self.all_indices, self.train_idx, self.val_idx, self.test_idx = [], [], [], []
        self.train_dataset, self.val_dataset, self.test_dataset, self.full_dataset = None, None, None, None

    def prepare_data(self):
        try:
            temp_dataset_for_counting = CarGraphDataset(self.root_dir, file_limit=self.file_limit)
            num_total_files = len(temp_dataset_for_counting)
            if self.file_limit is not None:
                # This print is fine, runs once in main process
                print(
                    f"Limiting dataset to {min(self.file_limit, num_total_files)} files (out of {num_total_files} found).")
                self.all_indices = list(range(min(self.file_limit, num_total_files)))
            else:
                self.all_indices = list(range(num_total_files))
        except Exception as e:
            print(f"Error determining total number of files in prepare_data: {e}")
            if self.file_limit:
                self.all_indices = list(range(self.file_limit))
            else:
                raise e

    def setup(self, stage=None):
        if not self.all_indices and len(self.all_indices) == 0: self.prepare_data()
        if not self.all_indices and self.file_limit is None:
            raise ValueError("Dataset indices could not be determined in setup. Check DATA_PATH.")
        elif not self.all_indices and self.file_limit is not None:
            print(f"Warning: No files found or accessible with file_limit={self.file_limit}. Datasets will be empty.")

        if not self.all_indices:
            print("Warning: all_indices is empty. Train/Val/Test datasets will be empty.")
            self.train_idx, self.val_idx, self.test_idx = [], [], []
        else:
            train_val_idx, self.test_idx = train_test_split(self.all_indices, test_size=0.15,
                                                            random_state=self.random_state)
            self.train_idx, self.val_idx = train_test_split(train_val_idx, test_size=0.15,
                                                            random_state=self.random_state)

        # These prints are fine, setup typically runs on main process before DDP fully starts or is handled by Lightning
        if stage == "fit" or stage is None:
            self.train_dataset = CarGraphDataset(self.root_dir, indices=self.train_idx)
            self.val_dataset = CarGraphDataset(self.root_dir, indices=self.val_idx)
            if self.train_dataset and self.val_dataset and len(self.train_dataset) > 0: print(
                f"Train dataset size: {len(self.train_dataset)}, Val dataset size: {len(self.val_dataset)}")
        if stage in ["test", "predict", None]:
            self.test_dataset = CarGraphDataset(self.root_dir, indices=self.test_idx)
            if self.test_dataset and len(self.test_dataset) > 0: print(f"Test dataset size: {len(self.test_dataset)}")

        self.full_dataset = CarGraphDataset(self.root_dir, indices=self.all_indices)
        if self.full_dataset and len(self.full_dataset) > 0: print(
            f"Full dataset size for analysis: {len(self.full_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset if self.train_dataset else [], batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset if self.val_dataset else [], batch_size=self.hparams.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset if self.test_dataset else [], batch_size=self.hparams.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def predict_dataloader(self):
        if self.test_dataset is None: self.setup(stage="test")
        return DataLoader(self.test_dataset if self.test_dataset else [], batch_size=self.hparams.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)


# === PyTorch Lightning System ===
class GraphAECDSystem(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, latent_dim, learning_rate=1e-4, weight_decay=1e-4,
                 cd_dropout_rate=0.0):
        super().__init__()
        self.save_hyperparameters("in_channels", "hidden_channels", "latent_dim", "learning_rate", "weight_decay",
                                  "cd_dropout_rate")
        self.model_ae = GraphAutoEncoder(self.hparams.in_channels, self.hparams.hidden_channels,
                                         self.hparams.latent_dim)
        self.model_cd = CdLinearModel(self.hparams.latent_dim, 1, dropout_rate=self.hparams.cd_dropout_rate)
        self.loss_weights = {"NodeRecon": 1.0, "CdRecon": 20.0, "ConnectRecon": 1.0}  # Matched to original notebook
        self.train_total_losses_epoch, self.val_total_losses_epoch = [], []
        self.train_cd_losses_epoch, self.val_cd_losses_epoch = [], []

    def forward(self, x, edge_index, batch_idx=None):
        return self.model_ae(x, edge_index, batch=batch_idx)

    def _shared_step(self, batch, batch_idx_loop):
        x, edge_index, b_idx = batch.x, batch.edge_index, batch.batch
        z_nodes, x_hat, z_graph = self(x, edge_index, batch_idx=b_idx)
        cd_pred = self.model_cd(z_graph)
        loss_x = MSELoss()(x_hat, x)
        cd_true = batch.cd_value.view(-1, 1)
        loss_cd = MSELoss()(cd_pred, cd_true)
        loss_a = torch.tensor(0.0, device=self.device)
        if self.loss_weights["ConnectRecon"] > 0 and edge_index.numel() > 0:  # Conditional computation
            z_i, z_j = z_nodes[edge_index[0]], z_nodes[edge_index[1]]
            dot_products = (z_i * z_j).sum(dim=1)
            adj_pred = torch.sigmoid(dot_products).clamp(min=1e-7, max=1 - 1e-7)
            adj_true = torch.ones_like(adj_pred);
            loss_a = BCELoss()(adj_pred, adj_true)
        total_loss = (self.loss_weights["NodeRecon"] * loss_x +
                      self.loss_weights["CdRecon"] * loss_cd +
                      self.loss_weights["ConnectRecon"] * loss_a)
        return total_loss, loss_x, loss_cd, loss_a, cd_pred, cd_true

    def training_step(self, batch, batch_idx_loop):
        total_loss, loss_x, loss_cd, loss_a, _, _ = self._shared_step(batch, batch_idx_loop)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch.num_graphs)
        self.log("train_cd_loss", loss_cd, on_step=False, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.log("train_node_loss", loss_x, on_step=False, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.log("train_adj_loss", loss_a, on_step=False, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        return total_loss

    def validation_step(self, batch, batch_idx_loop):
        total_loss, loss_x, loss_cd, loss_a, _, _ = self._shared_step(batch, batch_idx_loop)
        self.log("val_total_loss", total_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        self.log("val_cd_loss", loss_cd, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.log("val_node_loss", loss_x, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.log("val_adj_loss", loss_a, on_epoch=True, logger=True, batch_size=batch.num_graphs)

    def test_step(self, batch, batch_idx_loop):
        total_loss, loss_x, loss_cd, loss_a, cd_pred, cd_true = self._shared_step(batch, batch_idx_loop)
        self.log("test_total_loss", total_loss, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.log("test_cd_loss", loss_cd, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        return {"cd_pred": cd_pred, "cd_true": cd_true}

    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero: return  # Only collect and print on rank 0

        print(f"\n--- Rank 0: on_train_epoch_end (Epoch {self.current_epoch}) ---")
        metrics = self.trainer.callback_metrics
        print(f"Rank 0: Available Callback metrics keys: {list(metrics.keys())}")

        train_total_loss_val = metrics.get("train_total_loss_epoch")
        if train_total_loss_val is not None:
            self.train_total_losses_epoch.append(train_total_loss_val.item())
        else:
            print("Rank 0: train_total_loss_epoch NOT FOUND.")

        train_cd_loss_val = metrics.get("train_cd_loss")
        if train_cd_loss_val is not None:
            self.train_cd_losses_epoch.append(train_cd_loss_val.item())
        else:
            print("Rank 0: train_cd_loss NOT FOUND.")

        print(f"Rank 0: train_total_losses_epoch length: {len(self.train_total_losses_epoch)}")
        print(f"Rank 0: train_cd_losses_epoch length: {len(self.train_cd_losses_epoch)}")
        print(f"--- End Rank 0: on_train_epoch_end ---")

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero: return  # Only collect and print on rank 0

        print(f"\n--- Rank 0: on_validation_epoch_end (Epoch {self.current_epoch}) ---")
        if self.trainer.sanity_checking:
            print(
                f"Rank 0: Skipping metric collection for plotting due to sanity checking for epoch {self.current_epoch}.")
            print(f"--- End Rank 0: on_validation_epoch_end (sanity check) ---")
            return

        metrics = self.trainer.callback_metrics
        print(f"Rank 0: Available Callback metrics keys: {list(metrics.keys())}")

        val_total_loss_val = metrics.get("val_total_loss")
        if val_total_loss_val is not None:
            self.val_total_losses_epoch.append(val_total_loss_val.item())
        else:
            print("Rank 0: val_total_loss NOT FOUND.")

        val_cd_loss_val = metrics.get("val_cd_loss")
        if val_cd_loss_val is not None:
            self.val_cd_losses_epoch.append(val_cd_loss_val.item())
        else:
            print("Rank 0: val_cd_loss NOT FOUND.")

        print(f"Rank 0: val_total_losses_epoch length: {len(self.val_total_losses_epoch)}")
        print(f"Rank 0: val_cd_losses_epoch length: {len(self.val_cd_losses_epoch)}")
        # Note: Plotting logic moved to run_training after trainer.fit()
        print(f"--- End Rank 0: on_validation_epoch_end ---")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.model_ae.parameters()) + list(self.model_cd.parameters()),
            lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer


# === Plotting and Visualization Functions ===
def plot_losses_dual_static(train_losses, val_losses, cd_train_losses, cd_val_losses,
                            save_path="training_progress.png"):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    valid_lengths = [len(lst) for lst in [train_losses, val_losses, cd_train_losses, cd_val_losses] if
                     lst]  # Filter out empty lists before min
    if not valid_lengths: print("plot_losses_dual_static: All loss lists are empty for plotting."); plt.close(
        fig); return
    num_epoch_points = min(valid_lengths)
    if num_epoch_points == 0: print("plot_losses_dual_static: Min common length is 0. No plot generated."); plt.close(
        fig); return

    epochs = range(num_epoch_points)
    plot_data_defs = [
        (axs[0], train_losses, val_losses, "Autoencoder Total Loss", "Train Total", "Val Total", 'blue', 'orange'),
        (axs[1], cd_train_losses, cd_val_losses, "Cd Prediction Loss", "Train Cd", "Val Cd", 'green', 'red')]
    for ax, tr_d, vl_d, title, tr_l, vl_l, tr_c, vl_c in plot_data_defs:
        # Ensure lists have enough data before plotting and annotation
        can_plot_train = len(tr_d) >= num_epoch_points
        can_plot_val = len(vl_d) >= num_epoch_points

        if can_plot_train: ax.plot(epochs, tr_d[:num_epoch_points], label=tr_l, color=tr_c)
        if can_plot_val: ax.plot(epochs, vl_d[:num_epoch_points], label=vl_l, color=vl_c)

        ax.set_title(title);
        ax.set_xlabel("Epoch");
        ax.set_ylabel("Loss");
        ax.legend();
        ax.grid(True);
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if num_epoch_points > 0:  # Safe access for annotations
            if can_plot_train: ax.annotate(f"{tr_d[num_epoch_points - 1]:.6f}",
                                           (epochs[-1], tr_d[num_epoch_points - 1]), xytext=(-40, 0),
                                           textcoords="offset points", ha='center', color=tr_c)
            if can_plot_val: ax.annotate(f"{vl_d[num_epoch_points - 1]:.6f}", (epochs[-1], vl_d[num_epoch_points - 1]),
                                         xytext=(-40, -15), textcoords="offset points", ha='center', color=vl_c)

    title_ep = num_epoch_points - 1 if num_epoch_points > 0 else "N/A";
    plt.suptitle(f"Training Progress Summary (Epochs 0-{title_ep})")
    plt.tight_layout();
    Path(save_path).parent.mkdir(parents=True, exist_ok=True);
    plt.savefig(save_path);
    plt.close(fig)
    print(f"Summary loss plot saved to {save_path} (plotted {num_epoch_points} points)")


def plot_reconstruction(model_ae, dataset, device, sample_idx=None, max_nodes=100, save_dir="visualizations"):
    Path(save_dir).mkdir(parents=True, exist_ok=True);
    model_ae.eval()
    if not dataset or len(dataset) == 0: print("plot_reconstruction: Empty dataset."); return
    idx = random.choice(range(len(dataset))) if sample_idx is None else (
        sample_idx if 0 <= sample_idx < len(dataset) else 0)
    data = dataset[idx].to(device)
    if data.x is None or data.x.numel() == 0: print(
        f"plot_reconstruction: Sample #{idx} ({Path(data.filepath).name if hasattr(data, 'filepath') else 'N/A'}) has no node features."); return
    with torch.no_grad():
        _, x_hat, _ = model_ae(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long, device=device))
    x_orig, x_recon = data.x.cpu().numpy(), x_hat.cpu().numpy()
    num_nodes = x_orig.shape[0]
    if num_nodes == 0: print(f"plot_reconstruction: Sample #{idx} has no nodes after processing."); return
    sampled_indices = np.random.choice(num_nodes, min(max_nodes, num_nodes), replace=False)
    fig, axes = plt.subplots(1, min(3, x_orig.shape[1]), figsize=(15, 5), squeeze=False);
    axes = axes.flatten()
    for i, ax_i in enumerate(axes):
        ax_i.scatter(range(len(sampled_indices)), x_orig[sampled_indices, i], label="Original", alpha=0.7)
        ax_i.scatter(range(len(sampled_indices)), x_recon[sampled_indices, i], label="Reconstructed", alpha=0.7,
                     marker='x')
        ax_i.set_title(f"Feature {i}");
        ax_i.legend()
    fig_title_fp = Path(data.filepath).name if hasattr(data, 'filepath') else 'N/A'
    fig.suptitle(f"Scatter Comparison: Sample #{idx} ({fig_title_fp})");
    fig.tight_layout()
    save_path = Path(save_dir) / f"reconstruction_sample_{idx}.png";
    plt.savefig(save_path);
    plt.close(fig)
    print(f"Reconstruction plot saved to {save_path}")


def plot_cd_reconstructions(model_ae, model_cd, test_loader, device, save_dir="visualizations"):
    from sklearn.metrics import r2_score
    Path(save_dir).mkdir(parents=True, exist_ok=True);
    model_ae.eval();
    model_cd.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        if not test_loader or not hasattr(test_loader, 'dataset') or len(test_loader.dataset) == 0: print(
            "plot_cd_reconstructions: Empty loader."); return
        for i, batch in enumerate(test_loader):
            if batch is None or batch.x is None or batch.cd_value is None or batch.x.numel() == 0: print(
                f"plot_cd_reconstructions: Batch {i} invalid."); continue
            batch = batch.to(device);
            _, _, z_graph = model_ae(batch.x, batch.edge_index, batch.batch)
            all_preds.extend(model_cd(z_graph).view(-1).cpu().numpy());
            all_trues.extend(batch.cd_value.view(-1).cpu().numpy())
    if not all_trues or not all_preds: print("plot_cd_reconstructions: No data for R2 plot."); return
    all_trues_np, all_preds_np = np.array(all_trues), np.array(all_preds)
    r2 = r2_score(all_trues_np, all_preds_np) if len(all_trues_np) > 1 and len(all_preds_np) > 1 else float('nan')
    valid_indices = all_trues_np != 0
    rel_errors = np.abs(
        (all_preds_np[valid_indices] - all_trues_np[valid_indices]) / all_trues_np[valid_indices]) if np.any(
        valid_indices) else np.array([])
    max_rel_error = rel_errors.max() if rel_errors.size > 0 else float('nan');
    mean_rel_error = rel_errors.mean() if rel_errors.size > 0 else float('nan')
    print(f"Cd R²: {r2:.4f}, MaxRelErr: {max_rel_error:.6f}, MeanRelErr: {mean_rel_error:.6f}")
    plt.figure(figsize=(8, 6));
    plt.scatter(all_trues_np, all_preds_np, alpha=0.6, label="Predicted vs. True")
    min_val, max_val = (
        min(all_trues_np.min(), all_preds_np.min()) if len(all_trues_np) > 0 and len(all_preds_np) > 0 else 0), \
        (max(all_trues_np.max(), all_preds_np.max()) if len(all_trues_np) > 0 and len(all_preds_np) > 0 else 1)
    if min_val == max_val: max_val = min_val + (0.1 if max_val != 0 else 1.0)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label="Ideal")
    plt.xlabel("True Cd");
    plt.ylabel("Predicted Cd")
    r2_disp = f"{r2:.4f}" if not np.isnan(r2) else "N/A";
    max_rel_disp = f"{max_rel_error:.2%}" if not np.isnan(max_rel_error) else "N/A"
    plt.title(f"Cd: True vs Predicted\nR²={r2_disp}, MaxRelErr={max_rel_disp}")
    plt.grid(True);
    plt.legend();
    plt.tight_layout();
    save_path = Path(save_dir) / "cd_reconstruction_scatter.png";
    plt.savefig(save_path);
    plt.close()
    print(f"Cd scatter plot saved to {save_path}")


def save_latent_vectors(model_ae, dataset, device, output_path="latent_vectors.pt", batch_size=8):
    model_ae.eval()
    if not dataset or len(dataset) == 0: print("save_latent_vectors: Empty dataset."); return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_z_g, all_cd, all_fp = [], [], []
    print(f"Extracting latents for {len(dataset)} graphs...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None or batch.x is None or batch.x.numel() == 0: print(
                f"save_latent_vectors: Batch {i} invalid."); continue
            batch = batch.to(device);
            fp_curr = batch.filepath if hasattr(batch, 'filepath') else [f"NoFP_b{i}"]
            try:
                _, z_graph = model_ae.encode(x=batch.x, edge_index=batch.edge_index,
                                             batch=batch.batch)  # Use _ for unused z_nodes
                all_z_g.append(z_graph.cpu());
                all_cd.append(batch.cd_value.cpu());
                all_fp.extend(fp_curr)
            except Exception as e:
                print(f"ERROR processing batch {i} (files {fp_curr[:1]}): {e}. Skipping."); continue
            if (i + 1) % 10 == 0 or (i + 1) == len(loader): print(f"Processed batch {i + 1}/{len(loader)}")
    if not all_z_g: print("No latents extracted."); return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'latent_vectors': torch.cat(all_z_g), 'cd_values': torch.cat(all_cd), 'filepaths': all_fp}, output_path)
    print(f"Saved {len(all_fp)} latents to {output_path}")


def run_latent_optimization(model_ae, model_cd, dataset, device, target_cd=0.24, car_index=142, num_steps=600, lr=1e-4,
                            output_dir="optimized_geometry"):
    Path(output_dir).mkdir(parents=True, exist_ok=True);
    model_ae.eval();
    model_cd.eval()
    if not dataset or len(dataset) == 0 or car_index >= len(dataset): print(
        f"run_latent_opt: Invalid car_idx or empty dataset."); return None, None, None, None
    data = dataset[car_index].to(device)
    if data.x is None or data.x.numel() == 0: print(
        f"run_latent_opt: car {car_index} data empty."); return None, None, None, None
    s_batch = Batch.from_data_list([data]);
    x, ei, b_idx = s_batch.x, s_batch.edge_index, s_batch.batch
    with torch.no_grad():
        zn_orig, zg_orig = model_ae.encode(x, ei, b_idx); cd_orig = model_cd(zg_orig).item()
    print(
        f"Original Cd sample {car_index} ({Path(data.filepath).name if hasattr(data, 'filepath') else 'N/A'}): {cd_orig:.4f}")
    zn_opt = zn_orig.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([zn_opt], lr=lr)
    print(f"Optimizing for target Cd: {target_cd}...")
    for it in range(num_steps):
        zg_opt = global_mean_pool(zn_opt, b_idx);
        cd_pred = model_cd(zg_opt);
        loss = (cd_pred.squeeze() - target_cd).pow(2)
        opt.zero_grad();
        loss.backward();
        opt.step()
        if it % (num_steps // 10 or 1) == 0 or it == num_steps - 1: print(
            f"Iter {it:03d}: Loss={loss.item():.6f}, Cd_pred={cd_pred.item():.4f}")
    x_orig_nodes = x.cpu().numpy()
    with torch.no_grad():
        x_hat_opt_nodes = model_ae.decode(zn_opt).cpu().numpy(); cd_opt = model_cd(
            global_mean_pool(zn_opt, b_idx)).item()
    print(f"Optimization finished. Optimized Cd: {cd_opt:.4f}")
    center, scale = (data.center_point.cpu().numpy() if hasattr(data, 'center_point') else np.zeros(3)), \
        (data.scale.cpu().item() if hasattr(data, 'scale') else 1.0)
    x_orig_un, x_opt_un = x_orig_nodes[:, :3] * scale + center, x_hat_opt_nodes[:, :3] * scale + center
    fig = go.Figure();
    fig.add_trace(go.Scatter3d(x=x_orig_un[:, 0], y=x_orig_un[:, 1], z=x_orig_un[:, 2], mode='markers',
                               marker=dict(size=2, color='blue'), name=f'Original(Cd={cd_orig:.3f})'))
    fig.add_trace(go.Scatter3d(x=x_opt_un[:, 0], y=x_opt_un[:, 1], z=x_opt_un[:, 2], mode='markers',
                               marker=dict(size=2, color='red'), name=f'Optimized(Cd={cd_opt:.3f})'))
    fig.update_layout(title=f'Original vs. Optimized (Sample {car_index})',
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    plot_path = Path(output_dir) / f"optimized_geometry_comparison_sample_{car_index}.html"
    try:
        fig.write_html(str(plot_path), include_plotlyjs='cdn')
    except Exception as e:
        print(f"Error writing plotly HTML: {e}")
    print(f"Interactive 3D plot saved to {plot_path}")
    return x_orig_un, x_opt_un, ei.cpu(), (data.filepath if hasattr(data, 'filepath') else "dummy_filepath.npz")


def build_faces_from_edge_index(num_points, edge_index, max_faces=30000):
    edge_set = set();
    [[edge_set.add(tuple(sorted((edge_index[0, i].item(), edge_index[1, i].item())))) for i in
      range(edge_index.shape[1])]]
    faces = [];
    adj = [[] for _ in range(num_points)];
    [[adj[u].append(v), adj[v].append(u)] for u, v in edge_set]
    for i in range(num_points):
        for n1_idx in range(len(adj[i])):
            j = adj[i][n1_idx];
            if j <= i: continue
            for n2_idx in range(len(adj[i])):
                k = adj[i][n2_idx]
                if k <= j: continue
                if tuple(sorted((j, k))) in edge_set:
                    faces.append([i, j, k])
                    if len(faces) >= max_faces: print(f"Max faces ({max_faces}) hit."); return np.array(faces)
    return np.array(faces)


def export_point_cloud_and_mesh(points_np, edge_index_np, original_filepath, output_dir="optimized_geometry",
                                suffix="optimized"):
    Path(output_dir).mkdir(parents=True, exist_ok=True);
    base_fn = Path(original_filepath).stem
    pcd_fn = Path(output_dir) / f"{base_fn}_{suffix}_lineset.ply"
    pts_o3d = o3d.utility.Vector3dVector(points_np[:, :3]);
    lines_o3d = o3d.utility.Vector2iVector(edge_index_np.t().numpy())
    ls = o3d.geometry.LineSet(points=pts_o3d, lines=lines_o3d);
    ls.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(lines_o3d))])
    o3d.io.write_line_set(str(pcd_fn), ls);
    print(f"LineSet: {pcd_fn}")
    mesh_fn = Path(output_dir) / f"{base_fn}_{suffix}_mesh.obj";
    faces = build_faces_from_edge_index(len(points_np), edge_index_np)
    if faces.size > 0:
        mesh = o3d.geometry.TriangleMesh();
        mesh.vertices = pts_o3d;
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals();
        o3d.io.write_triangle_mesh(str(mesh_fn), mesh);
        print(f"Mesh: {mesh_fn}")
    else:
        print(f"No faces for {base_fn}_{suffix}.")


# === Main Execution Functions ===
def run_training(args, ckpt_to_resume=None):
    print("=== RUNNING TRAINING ===")
    pl.seed_everything(args.seed, workers=True);
    torch.set_float32_matmul_precision("high");
    torch.backends.cudnn.benchmark = True
    data_module = CarDataModule(root_dir=DATA_PATH, batch_size=args.batch_size, num_workers=args.num_dataloader_workers,
                                random_state=args.seed, file_limit=args.debug_file_limit)
    system = GraphAECDSystem(in_channels=args.in_channels, hidden_channels=args.hidden_channels,
                             latent_dim=args.latent_dim, learning_rate=args.learning_rate,
                             weight_decay=args.weight_decay, cd_dropout_rate=args.cd_dropout_rate)

    run_specific_dir_name = f"run_{args.experiment_name}_hc{args.hidden_channels}_ld{args.latent_dim}_lr{args.learning_rate}_wd{args.weight_decay}_cddr{args.cd_dropout_rate}_eps{args.max_epochs}"  # More descriptive
    checkpoint_dir = Path(CHECKPOINT_DIR_BASE) / run_specific_dir_name
    tensorboard_logger = TensorBoardLogger(save_dir=LOG_DIR_BASE, name=run_specific_dir_name, version="")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                          filename="best-ae-cd-{epoch:03d}-{val_total_loss:.4f}", save_top_k=1,
                                          monitor="val_total_loss", mode="min")
    progress_bar_callback = RichProgressBar()

    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices, callbacks=[checkpoint_callback, progress_bar_callback],
        logger=tensorboard_logger, log_every_n_steps=10,
        strategy=args.strategy if isinstance(args.devices, list) or (
                    isinstance(args.devices, int) and args.devices > 1) and torch.cuda.is_available() else "auto"
    )

    # Define effective_log_dir based on logger's properties
    if tensorboard_logger.version and isinstance(tensorboard_logger.version, str) and tensorboard_logger.version != "":
        effective_log_dir = Path(tensorboard_logger.save_dir) / tensorboard_logger.name / tensorboard_logger.version
    else:  # If version is empty or None, logs might be directly in save_dir/name
        effective_log_dir = Path(tensorboard_logger.save_dir) / tensorboard_logger.name
    effective_log_dir.mkdir(parents=True, exist_ok=True)  # Ensure log dir exists

    print(f"Starting training. Logs in: {effective_log_dir}")
    trainer.fit(system, datamodule=data_module, ckpt_path=ckpt_to_resume)
    print("Training finished.")

    plot_output_dir = effective_log_dir  # Use the specific versioned directory

    if system.train_total_losses_epoch and system.val_total_losses_epoch and \
            system.train_cd_losses_epoch and system.val_cd_losses_epoch:
        print(f"Generating final loss plot in: {plot_output_dir}")
        plot_save_sub_dir = plot_output_dir / "plots";
        plot_save_sub_dir.mkdir(parents=True, exist_ok=True)
        plot_losses_dual_static(system.train_total_losses_epoch, system.val_total_losses_epoch,
                                system.train_cd_losses_epoch, system.val_cd_losses_epoch,
                                plot_save_sub_dir / f"final_training_progress_epochs_{len(system.val_total_losses_epoch)}.png")
    else:
        print(
            f"Skipping final loss plot: lists empty. Lengths: TT={len(system.train_total_losses_epoch)}, VT={len(system.val_total_losses_epoch)}, TCD={len(system.train_cd_losses_epoch)}, VCD={len(system.val_cd_losses_epoch)}")

    if checkpoint_callback.best_model_path and Path(checkpoint_callback.best_model_path).exists():  # Check path exists
        print(f"Best model: {checkpoint_callback.best_model_path}");
        print("Testing best model...");
        trainer.test(system, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
    else:
        print("No best model checkpoint found or path invalid. Testing last model state."); trainer.test(system,
                                                                                                         datamodule=data_module)  # system is the last state
    print(
        f"Testing finished. Best checkpoint (if any during training): {checkpoint_callback.best_model_path if checkpoint_callback.best_model_path else 'N/A'}")


def run_analysis(args):
    print("=== RUNNING ANALYSIS ===")
    if not args.checkpoint_path or not Path(args.checkpoint_path).exists(): print(
        f"Error: Checkpoint '{args.checkpoint_path}' not found."); return
    pl.seed_everything(args.seed, workers=True);
    print(f"Loading model from: {args.checkpoint_path}")

    try:
        best_system = GraphAECDSystem.load_from_checkpoint(
            args.checkpoint_path,
            map_location=DEVICE,
            # Pass all hparams defined in GraphAECDSystem.__init__ signature that are also in args
            # This helps if checkpoint is from an older version or hparams were not perfectly saved
            in_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            cd_dropout_rate=args.cd_dropout_rate
        )
    except Exception as e:
        print(f"Error loading checkpoint with explicit hparams: {e}")
        print("Attempting to load checkpoint without explicit hparams (relying on saved hparams)...")
        try:
            best_system = GraphAECDSystem.load_from_checkpoint(args.checkpoint_path, map_location=DEVICE)
        except Exception as e_fallback:
            print(f"Fallback loading also failed: {e_fallback}. Cannot proceed with analysis.")
            return

    best_system.to(DEVICE);
    best_system.eval();
    model_ae, model_cd = best_system.model_ae, best_system.model_cd

    data_module = CarDataModule(root_dir=DATA_PATH, batch_size=args.batch_size, num_workers=args.num_dataloader_workers,
                                random_state=args.seed, file_limit=args.debug_file_limit)
    data_module.prepare_data();
    data_module.setup(stage="predict")

    checkpoint_p = Path(args.checkpoint_path)
    analysis_output_dir_base = checkpoint_p.parent / f"analysis_results_on_{checkpoint_p.stem}"
    analysis_output_dir_base.mkdir(parents=True, exist_ok=True);
    print(f"Analysis results in: {analysis_output_dir_base}")

    if data_module.test_dataset and len(data_module.test_dataset) > 0:
        print("Generating reconstruction plot...");
        plot_reconstruction(model_ae, data_module.test_dataset, DEVICE, sample_idx=0, save_dir=analysis_output_dir_base)
        print("Generating Cd prediction scatter plot...");
        plot_cd_reconstructions(model_ae, model_cd, data_module.test_dataloader(), DEVICE,
                                save_dir=analysis_output_dir_base)
    else:
        print("Skipping recon/Cd plots: test dataset empty.")

    if data_module.full_dataset and len(data_module.full_dataset) > 0:
        print("Saving latents...");
        save_latent_vectors(model_ae, data_module.full_dataset, DEVICE,
                            output_path=str(analysis_output_dir_base / "all_latent_vectors.pt"),
                            batch_size=args.batch_size)
        print("Running optimization & export...");
        optimization_output_dir = analysis_output_dir_base / "optimized_geometry_output"
        opt_car_idx = args.optimization_car_index
        if opt_car_idx >= len(data_module.full_dataset): print(
            f"Warn: car_idx {opt_car_idx} OOB. Using 0."); opt_car_idx = 0

        if len(data_module.full_dataset) > 0:
            opt_results = run_latent_optimization(model_ae, model_cd, data_module.full_dataset, DEVICE,
                                                  target_cd=args.optimization_target_cd, car_index=opt_car_idx,
                                                  num_steps=args.optimization_num_steps, lr=args.optimization_lr,
                                                  output_dir=optimization_output_dir)
            if opt_results and opt_results[1] is not None:
                x_orig, x_opt, edge_idx, fp = opt_results
                export_output_dir = optimization_output_dir
                print(f"Exporting OPTIMIZED: {Path(fp).name}");
                export_point_cloud_and_mesh(x_opt, edge_idx, fp, export_output_dir, "optimized")
                print(f"Exporting ORIGINAL: {Path(fp).name}");
                export_point_cloud_and_mesh(x_orig, edge_idx, fp, export_output_dir, "original")
            else:
                print("Skipping geometry export: optimization failed or invalid data.")
        else:
            print("Skipping optimization: full dataset empty.")
    else:
        print("Skipping latents/optimization: full dataset empty.")
    print("Analysis finished.")


def main():
    parser = argparse.ArgumentParser(description="Graph AE/CD Training and Analysis")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "analyze"], help="Run mode")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path for analysis or resuming train.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_dataloader_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--debug_file_limit", type=int, default=None, help="Limit files for debug")
    parser.add_argument("--experiment_name", type=str, default="default_exp", help="Experiment name for log/ckpt dirs")

    # Model hyperparameters matching original notebook for baseline
    parser.add_argument("--in_channels", type=int, default=10, help="AE input channels")
    parser.add_argument("--hidden_channels", type=int, default=64,
                        help="AE hidden channels (original notebook default)")
    parser.add_argument("--latent_dim", type=int, default=512, help="AE latent dimension (original notebook default)")
    parser.add_argument("--cd_dropout_rate", type=float, default=0.0,
                        help="Dropout rate for CdLinearModel (0.0 for original notebook baseline)")

    # Optimizer hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Optimizer weight decay (matched to original notebook)")

    # Training loop
    parser.add_argument("--max_epochs", type=int, default=300, help="Max training epochs")
    parser.add_argument("--devices", default=1,
                        help="Number of GPU devices (int) or list of IDs e.g. [0,1] or str 'auto'")  # Flexible devices
    parser.add_argument("--strategy", type=str, default="auto", help="Distributed training strategy")

    # Analysis specific
    parser.add_argument("--optimization_car_index", type=int, default=0, help="Car index for optimization")
    parser.add_argument("--optimization_target_cd", type=float, default=0.24, help="Target Cd for optimization")
    parser.add_argument("--optimization_num_steps", type=int, default=200, help="Optimization steps")
    parser.add_argument("--optimization_lr", type=float, default=1e-3, help="Optimization learning rate")
    args = parser.parse_args()

    # Process devices argument
    if isinstance(args.devices, str):
        if args.devices.lower() == "auto":
            args.devices = "auto"  # Let Lightning decide
        elif args.devices.isdigit() or (args.devices.startswith('-') and args.devices[1:].isdigit()):
            args.devices = int(args.devices)
        else:  # Assume list like "[0,1]"
            try:
                import ast
                args.devices = ast.literal_eval(args.devices)
                if not isinstance(args.devices, list): raise ValueError
            except:
                print(f"Warning: Could not parse devices='{args.devices}' as int or list. Defaulting to 1.")
                args.devices = 1

    ckpt_to_resume_fit = None
    if args.mode == "train" and args.checkpoint_path:  # Use --checkpoint_path for resuming
        if Path(args.checkpoint_path).exists():
            ckpt_to_resume_fit = args.checkpoint_path
            print(f"Attempting to resume training from: {ckpt_to_resume_fit}")
        else:
            print(
                f"Warning: Checkpoint path for resuming provided ('{args.checkpoint_path}') but not found. Starting new training.")

    if args.mode == "train":
        run_training(args, ckpt_to_resume=ckpt_to_resume_fit)
    elif args.mode == "analyze":
        if not args.checkpoint_path: parser.error("--checkpoint_path required for analysis.")
        run_analysis(args)
    else:
        parser.error(f"Invalid mode: {args.mode}. Choose 'train' or 'analyze'.")


if __name__ == '__main__':
    main()
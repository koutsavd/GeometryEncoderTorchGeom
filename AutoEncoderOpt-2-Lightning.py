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
import argparse  # Added for command-line arguments

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'png'  # or 'notebook_connected' or 'png' for static images

# === Configuration ===
DATA_PATH = "Graphs"
CHECKPOINT_DIR_BASE = "lightning_checkpoints"  # Base directory for checkpoints
LOG_DIR_BASE = "lightning_logs"  # Base directory for logs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Dataset Loader (from original notebook) ===
class CarGraphDataset(Dataset):
    def __init__(self, root_dir, indices=None, file_limit=None):
        self.root_dir = Path(root_dir)
        all_files_unsorted = [f for f in self.root_dir.rglob("*.npz") if "_adj" not in f.name]
        # Sort files numerically based on their names for consistency
        all_files_unsorted.sort(
            key=lambda f: int(f.stem.split('_')[-1]) if f.stem.split('_')[-1].isdigit() else float('inf'))

        if indices is not None:
            self.files = [all_files_unsorted[i] for i in indices if i < len(all_files_unsorted)]
        else:
            self.files = all_files_unsorted

        if file_limit is not None:
            self.files = self.files[:file_limit]

        super().__init__()

    def len(self):
        return len(self.files)

    def get(self, idx):
        feat_file = self.files[idx]
        adj_file = feat_file.with_name(feat_file.name.replace(".npz", "_adj.npz"))

        if not feat_file.exists() or not adj_file.exists():
            print(f"Warning: File not found for index {idx}, expected {feat_file}")
            return Data(x=torch.empty(0, 10), edge_index=torch.empty(2, 0, dtype=torch.long))

        npz_data = np.load(feat_file)

        x = torch.tensor(npz_data["x"], dtype=torch.float32)
        center_point = torch.tensor(npz_data["center_point"], dtype=torch.float32)
        scale = torch.tensor(npz_data["scale"], dtype=torch.float32)
        cd = torch.tensor(npz_data["cd"], dtype=torch.float32)

        a = sparse.load_npz(adj_file).tocoo()
        edge_index = torch.tensor(np.vstack((a.row, a.col)), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, center_point=center_point, cd_value=cd, scale=scale,
                    filepath=str(feat_file))


# === Models (from original notebook) ===
class CdLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = Sequential(
            Linear(in_features, 4 * in_features), ReLU(),
            Linear(4 * in_features, 2 * in_features), ReLU(),
            Linear(2 * in_features, in_features), ReLU(),
            Linear(in_features, out_features)
        )

    def forward(self, x):
        return self.model(x)


class GraphAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.res_proj = Linear(in_channels, latent_dim)
        self.encoder_convs = torch.nn.ModuleList([
            GATv2Conv(in_channels, hidden_channels),
            GATv2Conv(hidden_channels, 2 * hidden_channels // 4, heads=4, concat=True, residual=True),
            GATv2Conv(2 * hidden_channels, 4 * hidden_channels // 4, heads=4, concat=True, residual=True),
        ])
        self.encoder_norms = torch.nn.ModuleList([
            LayerNorm(hidden_channels),
            LayerNorm(2 * hidden_channels),
            LayerNorm(4 * hidden_channels),
        ])
        self.encoder_activations = torch.nn.ModuleList([ReLU(), ReLU(), ReLU()])

        self.encoder_lin = Sequential(
            Linear(4 * hidden_channels, latent_dim), ReLU(), Dropout(p=0.2)
        )
        self.decoder_lin = Sequential(
            Linear(latent_dim, 2 * hidden_channels), ReLU(),
            Linear(2 * hidden_channels, hidden_channels), ReLU(),
            Linear(hidden_channels, in_channels)
        )

    def encode(self, x, edge_index, batch=None):
        x_in = x
        for i, conv_layer in enumerate(self.encoder_convs):
            x = conv_layer(x, edge_index)
            x = self.encoder_norms[i](x)
            x = self.encoder_activations[i](x)

        z_nodes = self.encoder_lin(x)
        z_nodes = z_nodes + self.res_proj(x_in)  # Residual connection

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        z_graph = global_mean_pool(z_nodes, batch)
        return z_nodes, z_graph

    def decode(self, z_nodes):
        return self.decoder_lin(z_nodes)

    def forward(self, x, edge_index, batch=None):
        z_nodes, z_graph = self.encode(x, edge_index, batch)
        x_hat = self.decode(z_nodes)
        return z_nodes, x_hat, z_graph


# === PyTorch Lightning DataModule ===
class CarDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, num_workers=4, random_state=42, file_limit=None):
        super().__init__()
        self.save_hyperparameters()  # Saves args to self.hparams (batch_size, etc.)
        self.root_dir = root_dir
        # self.batch_size = batch_size # Already saved by save_hyperparameters
        self.num_workers = num_workers
        self.random_state = random_state
        self.file_limit = file_limit
        self.all_indices = []
        self.train_idx, self.val_idx, self.test_idx = [], [], []
        self.train_dataset, self.val_dataset, self.test_dataset, self.full_dataset = None, None, None, None

    def prepare_data(self):
        try:
            temp_dataset_for_counting = CarGraphDataset(self.root_dir, file_limit=self.file_limit)
            num_total_files = len(temp_dataset_for_counting)
            if self.file_limit is not None:
                print(f"Using a limited dataset of {self.file_limit} files.")
            self.all_indices = list(range(num_total_files))
        except Exception as e:
            print(f"Error determining total number of files in prepare_data: {e}")
            if self.file_limit:
                self.all_indices = list(range(self.file_limit))
            else:
                raise e

    def setup(self, stage=None):
        if not self.all_indices:
            self.prepare_data()
        if not self.all_indices and self.file_limit is None:  # If still no indices and not a debug limit
            raise ValueError("Dataset indices could not be determined in setup. Check DATA_PATH and file structure.")
        elif not self.all_indices and self.file_limit is not None:  # If limited and still no indices (e.g. limit > actual files)
            # This case should ideally be caught by prepare_data if len(temp_dataset) was 0
            # However, if prepare_data used file_limit to make self.all_indices, but it's empty.
            print(f"Warning: No files found even with file_limit={self.file_limit}. Proceeding with empty dataset.")
            # This will likely lead to errors downstream in DataLoader if dataset is truly empty.

        # Use hparams for batch_size as it's saved by save_hyperparameters()
        current_batch_size = self.hparams.batch_size

        train_val_idx, self.test_idx = train_test_split(self.all_indices, test_size=0.15,
                                                        random_state=self.random_state)
        self.train_idx, self.val_idx = train_test_split(train_val_idx, test_size=0.15, random_state=self.random_state)

        if stage == "fit" or stage is None:
            self.train_dataset = CarGraphDataset(self.root_dir, indices=self.train_idx,
                                                 file_limit=self.file_limit if self.file_limit and not self.train_idx else None)
            self.val_dataset = CarGraphDataset(self.root_dir, indices=self.val_idx,
                                               file_limit=self.file_limit if self.file_limit and not self.val_idx else None)
            if self.train_dataset and self.val_dataset:
                print(f"Train dataset size: {len(self.train_dataset)}, Val dataset size: {len(self.val_dataset)}")
        if stage == "test" or stage == "predict" or stage is None:  # Added predict for analysis stage
            self.test_dataset = CarGraphDataset(self.root_dir, indices=self.test_idx,
                                                file_limit=self.file_limit if self.file_limit and not self.test_idx else None)
            if self.test_dataset:
                print(f"Test dataset size: {len(self.test_dataset)}")

        self.full_dataset = CarGraphDataset(self.root_dir, indices=self.all_indices, file_limit=self.file_limit)
        if self.full_dataset:
            print(f"Full dataset size: {len(self.full_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.num_workers,
                          persistent_workers=True if self.num_workers > 0 else False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.num_workers,
                          persistent_workers=True if self.num_workers > 0 else False)

    def predict_dataloader(self):  # For analysis, might use test_dataset or full_dataset
        # Often, for analysis tasks like latent vector saving or specific visualizations,
        # you might want to iterate over a specific dataset (e.g., full_dataset without shuffle).
        # Here, we provide the test_loader as a default predict_loader.
        # You can customize this if needed for specific analysis tasks.
        if self.test_dataset is None: self.setup(stage="test")  # Ensure test_dataset is available
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.num_workers,
                          persistent_workers=True if self.num_workers > 0 else False)


# === PyTorch Lightning System ===
class GraphAECDSystem(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, latent_dim, learning_rate=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model_ae = GraphAutoEncoder(in_channels, hidden_channels, latent_dim)
        self.model_cd = CdLinearModel(latent_dim, 1)

        self.loss_weights = {"NodeRecon": 1.0, "CdRecon": 20.0, "ConnectRecon": 1.0}

        self.train_losses_epoch = []
        self.val_losses_epoch = []
        self.cd_train_losses_epoch = []
        self.cd_val_losses_epoch = []

    def forward(self, x, edge_index, batch_idx=None):
        return self.model_ae(x, edge_index, batch=batch_idx)

    def _shared_step(self, batch, batch_idx_loop):
        x, edge_index, b_idx = batch.x, batch.edge_index, batch.batch

        z_nodes, x_hat, z_graph = self(x, edge_index, batch_idx=b_idx)
        cd_pred = self.model_cd(z_graph)

        loss_x = MSELoss()(x_hat, x)
        cd_true = batch.cd_value.view(-1, 1)
        loss_cd = MSELoss()(cd_pred, cd_true)

        loss_a = torch.tensor(0.0, device=self.device)  # Default to 0 if no edges
        if edge_index.numel() > 0:  # Check if there are any edges
            z_i = z_nodes[edge_index[0]]
            z_j = z_nodes[edge_index[1]]
            dot_products = (z_i * z_j).sum(dim=1)
            adj_pred = torch.sigmoid(dot_products).clamp(min=1e-7, max=1 - 1e-7)
            adj_true = torch.ones_like(adj_pred)
            loss_a = BCELoss()(adj_pred, adj_true)

        total_loss = (self.loss_weights["NodeRecon"] * loss_x +
                      self.loss_weights["CdRecon"] * loss_cd +
                      self.loss_weights["ConnectRecon"] * loss_a)

        return total_loss, loss_x, loss_cd, loss_a, cd_pred, cd_true

    def training_step(self, batch, batch_idx_loop):
        total_loss, loss_x, loss_cd, loss_a, _, _ = self._shared_step(batch, batch_idx_loop)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch.num_graphs)
        self.log("train_node_loss", loss_x, on_step=False, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.log("train_cd_loss", loss_cd, on_step=False, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.log("train_adj_loss", loss_a, on_step=False, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        return total_loss

    def validation_step(self, batch, batch_idx_loop):
        total_loss, loss_x, loss_cd, loss_a, _, _ = self._shared_step(batch, batch_idx_loop)
        self.log("val_total_loss", total_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        self.log("val_node_loss", loss_x, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.log("val_cd_loss", loss_cd, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.log("val_adj_loss", loss_a, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        return total_loss

    def test_step(self, batch, batch_idx_loop):
        total_loss, loss_x, loss_cd, loss_a, cd_pred, cd_true = self._shared_step(batch, batch_idx_loop)
        self.log("test_total_loss", total_loss, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.log("test_node_loss", loss_x, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.log("test_cd_loss", loss_cd, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        self.log("test_adj_loss", loss_a, on_epoch=True, logger=True, batch_size=batch.num_graphs)
        return {"cd_pred": cd_pred, "cd_true": cd_true}

    def on_train_epoch_end(self):
        if "train_total_loss_epoch" in self.trainer.callback_metrics:
            self.train_losses_epoch.append(self.trainer.callback_metrics["train_total_loss_epoch"].item())
        if "train_cd_loss_epoch" in self.trainer.callback_metrics:
            self.cd_train_losses_epoch.append(self.trainer.callback_metrics["train_cd_loss_epoch"].item())

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            if "val_total_loss" in self.trainer.callback_metrics:
                self.val_losses_epoch.append(self.trainer.callback_metrics["val_total_loss"].item())
            if "val_cd_loss" in self.trainer.callback_metrics:
                self.cd_val_losses_epoch.append(self.trainer.callback_metrics["val_cd_loss"].item())

            if self.train_losses_epoch and self.val_losses_epoch and \
                    self.cd_train_losses_epoch and self.cd_val_losses_epoch:
                # Ensure the logger's save_dir is used for placing these plots
                plot_save_dir = Path(self.trainer.logger.save_dir) / "plots"
                plot_save_dir.mkdir(parents=True, exist_ok=True)
                plot_losses_dual_static(
                    self.train_losses_epoch, self.val_losses_epoch,
                    self.cd_train_losses_epoch, self.cd_val_losses_epoch,
                    plot_save_dir / f"training_progress_epoch_{self.current_epoch}.png"
                )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.model_ae.parameters()) + list(self.model_cd.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer


# === Plotting and Visualization Functions (adapted from original) ===
# (plot_losses_dual_static, plot_reconstruction, plot_cd_reconstructions, save_latent_vectors,
#  run_latent_optimization, build_faces_from_edge_index, export_point_cloud_and_mesh remain mostly the same.
#  Ensure save paths within these functions correctly use the output directories passed to them.)

def plot_losses_dual_static(train_losses, val_losses, cd_train_losses, cd_val_losses,
                            save_path="training_progress.png"):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(len(train_losses))
    if not epochs:  # No data to plot
        plt.close(fig)
        print("No data to plot for losses yet.")
        return

    # --- Plot total loss ---
    axs[0].plot(epochs, train_losses, label="Train Total Loss", color='blue')
    axs[0].plot(epochs, val_losses, label="Val Total Loss", color='orange')
    axs[0].set_title("Autoencoder Total Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    if train_losses: axs[0].annotate(f"{train_losses[-1]:.6f}", (epochs[-1], train_losses[-1]),
                                     textcoords="offset points", xytext=(-40, 0), ha='center', color='blue')
    if val_losses: axs[0].annotate(f"{val_losses[-1]:.6f}", (epochs[-1], val_losses[-1]), textcoords="offset points",
                                   xytext=(-40, -15), ha='center', color='orange')

    # --- Plot Cd loss ---
    axs[1].plot(epochs, cd_train_losses, label="Train Cd Loss", color='green')
    axs[1].plot(epochs, cd_val_losses, label="Val Cd Loss", color='red')
    axs[1].set_title("Cd Prediction Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    if cd_train_losses: axs[1].annotate(f"{cd_train_losses[-1]:.6f}", (epochs[-1], cd_train_losses[-1]),
                                        textcoords="offset points", xytext=(-40, 0), ha='center', color='green')
    if cd_val_losses: axs[1].annotate(f"{cd_val_losses[-1]:.6f}", (epochs[-1], cd_val_losses[-1]),
                                      textcoords="offset points", xytext=(-40, -15), ha='center', color='red')

    plt.suptitle(f"Training Progress (Epoch {len(epochs)})")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Loss plot saved to {save_path}")


def plot_reconstruction(model_ae, dataset, device, sample_idx=None, max_nodes=100, save_dir="visualizations"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model_ae.eval()
    idx = random.choice(range(len(dataset))) if sample_idx is None else sample_idx
    data = dataset[idx].to(device)

    with torch.no_grad():
        _, x_hat, _ = model_ae(
            data.x, data.edge_index,
            torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        )

    x_orig = data.x.cpu().numpy()
    x_recon = x_hat.cpu().numpy()

    num_nodes = x_orig.shape[0]
    sampled_indices = np.random.choice(num_nodes, min(max_nodes, num_nodes), replace=False)

    fig, axes = plt.subplots(1, min(3, x_orig.shape[1]), figsize=(15, 5))
    if min(3, x_orig.shape[1]) == 1: axes = [axes]

    for i, ax_i in enumerate(axes):
        ax_i.scatter(range(len(sampled_indices)), x_orig[sampled_indices, i], label="Original", alpha=0.7)
        ax_i.scatter(range(len(sampled_indices)), x_recon[sampled_indices, i], label="Reconstructed", alpha=0.7,
                     marker='x')
        ax_i.set_title(f"Feature {i}")
        ax_i.legend()

    fig.suptitle(f"Scatter Comparison: Sample #{idx} ({Path(data.filepath).name})")
    fig.tight_layout()
    save_path = Path(save_dir) / f"reconstruction_sample_{idx}.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Reconstruction plot saved to {save_path}")


def plot_cd_reconstructions(model_ae, model_cd, test_loader, device, save_dir="visualizations"):
    from sklearn.metrics import r2_score
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model_ae.eval()
    model_cd.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        if not test_loader or len(test_loader.dataset) == 0: # Check if dataset is empty
            print("Warning: test_loader or its dataset is empty in plot_cd_reconstructions.")
        for i, batch in enumerate(test_loader): # Add an iterator count
            print(f"Processing batch {i} in plot_cd_reconstructions...") # Debug print
            if batch is None or batch.x is None or batch.cd_value is None: # Check for valid batch
                print(f"Warning: Batch {i} is invalid or missing data.")
                continue
            batch = batch.to(device)
            x, edge_index, b_idx = batch.x, batch.edge_index, batch.batch
            _, _, z_graph = model_ae(x, edge_index, batch=b_idx)
            cd_pred_batch = model_cd(z_graph).view(-1).cpu().numpy()
            cd_true_batch = batch.cd_value.view(-1).cpu().numpy()
            all_preds.extend(cd_pred_batch)
            all_trues.extend(cd_true_batch)

    if not all_trues or not all_preds:
        print("Error: No true or predicted values were collected. Cannot calculate R2 or plot.")
        # Optionally, create a placeholder plot or just return
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No data to plot for Cd reconstructions.", ha='center', va='center')
        plt.title("Cd Prediction: True vs Predicted (No Data)")
        save_path = Path(save_dir) / "cd_reconstruction_scatter_nodata.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Placeholder Cd reconstruction plot saved to {save_path}")
        return


    all_trues_np = np.array(all_trues)
    all_preds_np = np.array(all_preds)

    # --- START DEBUG PRINTS ---
    print(f"\n--- Debug Info for plot_cd_reconstructions ---")
    print(f"Shape of all_trues_np: {all_trues_np.shape}")
    print(f"Content of all_trues_np (first 5): {all_trues_np[:5]}")
    print(f"Shape of all_preds_np: {all_preds_np.shape}")
    print(f"Content of all_preds_np (first 5): {all_preds_np[:5]}")

    r2 = float('nan') # Default if calculation fails
    try:
        if len(all_trues_np) > 0 and len(all_preds_np) > 0 and len(all_trues_np) == len(all_preds_np):
            r2 = r2_score(all_trues_np, all_preds_np)
        else:
            print("Warning: Cannot compute R2 score due to empty or mismatched arrays.")
    except Exception as e:
        print(f"Error computing R2 score: {e}")
        # r2 remains float('nan')

    print(f"Value of r2: {r2}, type: {type(r2)}")

    valid_indices = all_trues_np != 0
    rel_errors = np.array([])
    if np.any(valid_indices) and len(all_preds_np[valid_indices]) == len(all_trues_np[valid_indices]): # Added length check
        rel_errors = np.abs((all_preds_np[valid_indices] - all_trues_np[valid_indices]) / all_trues_np[valid_indices])

    max_rel_error = rel_errors.max() if len(rel_errors) > 0 else float('nan')
    mean_rel_error = rel_errors.mean() if len(rel_errors) > 0 else float('nan')
    print(f"Value of max_rel_error: {max_rel_error}, type: {type(max_rel_error)}")
    print(f"Value of mean_rel_error: {mean_rel_error}, type: {type(mean_rel_error)}")
    print(f"--- End Debug Info ---\n")
    # --- END DEBUG PRINTS ---

    r2_display = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
    max_rel_error_display = f"{max_rel_error:.2%}" if not np.isnan(max_rel_error) else "N/A"


    plt.figure(figsize=(8, 6))
    plt.scatter(all_trues_np, all_preds_np, alpha=0.6, label="Predicted vs. True")
    # Ensure min_val/max_val don't fail on empty arrays
    min_val = 0
    max_val = 1
    if len(all_trues_np) > 0 and len(all_preds_np) > 0:
        min_val = min(min(all_trues_np), min(all_preds_np))
        max_val = max(max(all_trues_np), max(all_preds_np))
    elif len(all_trues_np) > 0:
        min_val = min(all_trues_np)
        max_val = max(all_trues_np)
    elif len(all_preds_np) > 0:
        min_val = min(all_preds_np)
        max_val = max(all_preds_np)

    if min_val == max_val: # Avoid plotting a single point for the ideal line if all values are same
         max_val = min_val + 1


    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label="Ideal")
    plt.xlabel("True Cd")
    plt.ylabel("Predicted Cd")
    # Original problematic line:
    # plt.title(
    #     f"Cd Prediction: True vs Predicted\nR²={r2:.4f}, Max Rel Error={max_rel_error:.2% if not np.isnan(max_rel_error) else 'N/A'}"
    # )
    # Safer title formatting:
    plt.title(
        f"Cd Prediction: True vs Predicted\nR²={r2_display}, Max Rel Error={max_rel_error_display}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path = Path(save_dir) / "cd_reconstruction_scatter.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Cd reconstruction scatter plot saved to {save_path}")

def save_latent_vectors(model_ae, dataset, device, output_path="latent_vectors.pt", batch_size=8):
    model_ae.eval()
    # It's good to ensure the dataset for the loader is not empty.
    if not dataset or len(dataset) == 0:
        print(f"Error in save_latent_vectors: The provided dataset is empty or None.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_z_graphs, all_cds, all_filepaths = [], [], []

    print(f"Starting latent vector extraction for {len(dataset)} graphs...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"\n--- Processing Batch {i} in save_latent_vectors ---") # Indent this print to be inside the loop
            if batch is None:
                print(f"Warning: Batch {i} is None. Skipping.")
                continue

            batch = batch.to(device)
            x, edge_index, b_idx = batch.x, batch.edge_index, batch.batch
            current_filepaths = batch.filepath if hasattr(batch, 'filepath') else [f"Filepath_N/A_for_batch_{i}"]

            # --- START DEBUG PRINTS ---
            #print(f"Batch object type: {type(batch)}")
            #print(f"x.shape: {x.shape}, x.dtype: {x.dtype}, x.device: {x.device}")
            #print(f"edge_index.shape: {edge_index.shape}, edge_index.dtype: {edge_index.dtype}")
            #if x.shape[0] > 0 and edge_index.numel() > 0: # Check if there are nodes and edges before min/max
            #    print(f"edge_index min: {edge_index.min()}, edge_index max: {edge_index.max()} vs num_nodes: {x.shape[0]}")
            #elif x.shape[0] == 0:
            #    print("Warning: x (node features) is empty for this batch.")

            #print(f"b_idx (batch.batch) shape: {b_idx.shape if b_idx is not None else 'None'}, "
            #      f"dtype: {b_idx.dtype if b_idx is not None else 'N/A'}")
            #print(f"batch.num_graphs: {batch.num_graphs if hasattr(batch, 'num_graphs') else 'N/A'}")

            #current_filepaths = batch.filepath if hasattr(batch, 'filepath') else ["N/A"]
            #print(f"Filepaths in this batch (first few): {current_filepaths[:min(3, len(current_filepaths))]}")
            #print(f"--- End Debug Info for Batch {i} ---\n")
            # --- END DEBUG PRINTS ---

            try:
                # This is the line causing the error in the traceback
                    _, z_graph = model_ae.encode(x, edge_index, batch=b_idx)

                    all_z_graphs.append(z_graph.cpu())
                    all_cds.append(batch.cd_value.cpu())
                    all_filepaths.extend(current_filepaths)
            except Exception as e:
                print(f"ERROR occurred while processing batch {i}, containing files (e.g.): {current_filepaths[:min(3, len(current_filepaths))]}")
                print(f"Specific error: {e}")
                # Optionally re-raise the exception or continue to see if other batches work
                # raise e # Uncomment to stop on the first erroring batch
                print("Skipping this batch due to error.")
                continue # Try to process next batches

            if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                print(f"Processed batch {i+1}/{len(loader)} for latent vector extraction (current z_graphs collected: {len(all_z_graphs)})")

    if not all_z_graphs:
        print("No latent vectors were successfully extracted.")
        return

    all_z_graphs_tensor = torch.cat(all_z_graphs, dim=0)
    all_cds_tensor = torch.cat(all_cds, dim=0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'latent_vectors': all_z_graphs_tensor, 'cd_values': all_cds_tensor, 'filepaths': all_filepaths}, output_path)
    print(f"Saved {all_z_graphs_tensor.shape[0]} latent vectors to {output_path}")

def run_latent_optimization(model_ae, model_cd, dataset, device, target_cd=0.24, car_index=142, num_steps=600, lr=1e-4,
                            output_dir="optimized_geometry"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_ae.eval()
    model_cd.eval()

    if car_index >= len(dataset):
        print(f"Error: car_index {car_index} is out of bounds for dataset of size {len(dataset)}.")
        return None, None, None, None

    data = dataset[car_index].to(device)
    single_graph_batch = Batch.from_data_list([data])
    x, edge_index, b_idx = single_graph_batch.x, single_graph_batch.edge_index, single_graph_batch.batch

    with torch.no_grad():
        z_nodes_orig, z_graph_orig = model_ae.encode(x, edge_index, batch=b_idx)
        cd_original = model_cd(z_graph_orig).item()
    print(f"Original Cd for sample {car_index} ({Path(data.filepath).name}): {cd_original:.4f}")

    z_nodes_opt = z_nodes_orig.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z_nodes_opt], lr=lr)

    print(f"Optimizing latent space for target Cd: {target_cd}...")
    for it in range(num_steps):
        z_graph_opt = global_mean_pool(z_nodes_opt, b_idx)
        cd_pred = model_cd(z_graph_opt)
        loss = (cd_pred.squeeze() - target_cd).pow(2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % (num_steps // 10 or 1) == 0 or it == num_steps - 1:  # Ensure modulo is not by zero
            print(f"Iter {it:03d}: Loss = {loss.item():.6f}, Cd_pred = {cd_pred.item():.4f}")

    x_original_nodes = x.cpu().numpy()
    with torch.no_grad():
        x_hat_optimized_nodes = model_ae.decode(z_nodes_opt).cpu().numpy()
        cd_optimized = model_cd(global_mean_pool(z_nodes_opt, b_idx)).item()
    print(f"Optimization finished. Optimized Cd: {cd_optimized:.4f}")

    center = data.center_point.cpu().numpy()
    scale = data.scale.cpu().item()

    x_original_unnorm = x_original_nodes[:, :3] * scale + center
    x_optimized_unnorm = x_hat_optimized_nodes[:, :3] * scale + center

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_original_unnorm[:, 0], y=x_original_unnorm[:, 1], z=x_original_unnorm[:, 2],
                               mode='markers', marker=dict(size=2, color='blue'),
                               name=f'Original (Cd={cd_original:.3f})'))
    fig.add_trace(go.Scatter3d(x=x_optimized_unnorm[:, 0], y=x_optimized_unnorm[:, 1], z=x_optimized_unnorm[:, 2],
                               mode='markers', marker=dict(size=2, color='red'),
                               name=f'Optimized (Cd={cd_optimized:.3f})'))
    fig.update_layout(title=f'Original vs. Optimized Geometry (Sample {car_index})',
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

    plot_path = Path(output_dir) / f"optimized_geometry_comparison_sample_{car_index}.html"
    fig.write_html(str(plot_path))
    print(f"Interactive 3D plot saved to {plot_path}")

    return x_original_unnorm, x_optimized_unnorm, edge_index.cpu(), data.filepath


def build_faces_from_edge_index(num_points, edge_index, max_faces=30000):
    edge_set = set()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        edge_set.add(tuple(sorted((u, v))))

    faces = []
    adj = [[] for _ in range(num_points)]
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(num_points):
        for neighbor1_idx in range(len(adj[i])):
            j = adj[i][neighbor1_idx]
            if j <= i: continue
            for neighbor2_idx in range(len(adj[i])):  # Iterate through neighbors of i again
                k = adj[i][neighbor2_idx]
                if k <= j: continue

                if tuple(sorted((j, k))) in edge_set:
                    faces.append([i, j, k])
                    if len(faces) >= max_faces:
                        print(f"Reached max_faces limit ({max_faces}) for mesh generation.")
                        return np.array(faces)
    return np.array(faces)


def export_point_cloud_and_mesh(points_np, edge_index_np, original_filepath, output_dir="optimized_geometry",
                                suffix="optimized"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base_filename = Path(original_filepath).stem

    pcd_filename = Path(output_dir) / f"{base_filename}_{suffix}_lineset.ply"
    points_o3d = o3d.utility.Vector3dVector(points_np[:, :3])
    lines_o3d = o3d.utility.Vector2iVector(edge_index_np.t().numpy())

    line_set = o3d.geometry.LineSet(points=points_o3d, lines=lines_o3d)
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(lines_o3d))])
    o3d.io.write_line_set(str(pcd_filename), line_set)
    print(f"LineSet saved to {pcd_filename}")

    mesh_filename = Path(output_dir) / f"{base_filename}_{suffix}_mesh.obj"
    faces_np = build_faces_from_edge_index(len(points_np), edge_index_np)

    if faces_np.size > 0:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = points_o3d
        mesh.triangles = o3d.utility.Vector3iVector(faces_np)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(mesh_filename), mesh)
        print(f"Mesh saved to {mesh_filename}")
    else:
        print(f"No faces generated for mesh from {base_filename}_{suffix}. Skipping mesh export.")


# === Main Execution Functions ===
def run_training(args):
    print("=== RUNNING TRAINING ===")
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # --- DataModule ---
    data_module = CarDataModule(
        root_dir=DATA_PATH,
        batch_size=args.batch_size,
        num_workers=args.num_dataloader_workers,
        random_state=args.seed,
        file_limit=args.debug_file_limit
    )

    # --- LightningModule ---
    system = GraphAECDSystem(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # --- Callbacks ---
    # Create a unique directory for this run's checkpoints and logs
    run_specific_dir_name = f"run_{args.experiment_name}_seed{args.seed}"
    checkpoint_dir = Path(CHECKPOINT_DIR_BASE) / run_specific_dir_name
    log_dir = Path(LOG_DIR_BASE) / run_specific_dir_name

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-graph-ae-cd-{epoch:03d}-{val_total_loss:.4f}",
        save_top_k=1,
        monitor="val_total_loss",
        mode="min"
    )
    progress_bar_callback = RichProgressBar()

    # --- Logger ---
    tensorboard_logger = TensorBoardLogger(save_dir=LOG_DIR_BASE, name=run_specific_dir_name,
                                           version="")  # Empty version to use name directly

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        callbacks=[checkpoint_callback, progress_bar_callback],
        logger=tensorboard_logger,
        log_every_n_steps=10,
        strategy=args.strategy if args.devices != 1 and torch.cuda.is_available() and args.devices > 1 else "auto",
    )

    print(f"Starting training. Logs and checkpoints will be saved in: {log_dir.parent}")
    trainer.fit(system, datamodule=data_module,
                ckpt_path=args.resume_checkpoint_path if args.resume_checkpoint_path else None)
    print("Training finished.")

    print("Starting testing with the best model from training...")
    if checkpoint_callback.best_model_path:
        print(f"Best model path: {checkpoint_callback.best_model_path}")
        trainer.test(system, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
    else:
        print("No best model checkpoint found from training. Testing with the last model state.")
        trainer.test(system, datamodule=data_module)
    print("Testing finished.")
    print(f"Path to best checkpoint: {checkpoint_callback.best_model_path}")


def run_analysis(args):
    print("=== RUNNING ANALYSIS ===")
    if not args.checkpoint_path or not Path(args.checkpoint_path).exists():
        print(f"Error: Checkpoint path '{args.checkpoint_path}' not provided or does not exist.")
        return

    pl.seed_everything(args.seed, workers=True)  # For reproducibility in dataset sampling etc.

    # --- Load LightningModule from checkpoint ---
    # Hyperparameters like in_channels, hidden_channels, latent_dim are loaded from the checkpoint
    # if they were saved with save_hyperparameters() in the LightningModule.
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    # Pass hparams_file=None if you are sure all necessary hparams are in the checkpoint
    # or if you will manually provide them to the model class.
    # GraphAECDSystem.load_from_checkpoint internally handles hparams.
    best_system = GraphAECDSystem.load_from_checkpoint(args.checkpoint_path, map_location=DEVICE)
    best_system.to(DEVICE)
    best_system.eval()

    model_ae = best_system.model_ae
    model_cd = best_system.model_cd

    # --- DataModule for analysis ---
    # We need the DataModule to get datasets for visualization, etc.
    # Use batch_size from args for the dataloaders in analysis.
    data_module = CarDataModule(
        root_dir=DATA_PATH,
        batch_size=args.batch_size,  # Use current batch_size arg for analysis dataloaders
        num_workers=args.num_dataloader_workers,
        random_state=args.seed,
        file_limit=args.debug_file_limit
    )
    data_module.prepare_data()  # Ensure all_indices is populated
    data_module.setup(stage="predict")  # Setup test_dataset and full_dataset

    # --- Output directory for analysis results ---
    # Place results in a subfolder of the checkpoint's directory or a new analysis dir
    checkpoint_p = Path(args.checkpoint_path)
    analysis_output_dir = checkpoint_p.parent / f"analysis_results_on_{checkpoint_p.stem}"
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analysis results will be saved in: {analysis_output_dir}")

    # --- Run Post-Training Analysis tasks ---
    if data_module.test_dataset and len(data_module.test_dataset) > 0:
        print("Generating reconstruction plot...")
        plot_reconstruction(model_ae, data_module.test_dataset, DEVICE, sample_idx=0, save_dir=analysis_output_dir)

        print("Generating Cd prediction scatter plot...")
        plot_cd_reconstructions(model_ae, model_cd, data_module.test_dataloader(), DEVICE, save_dir=analysis_output_dir)
    else:
        print("Skipping reconstruction and Cd scatter plots as test dataset is empty or unavailable.")

    if data_module.full_dataset and len(data_module.full_dataset) > 0:
        print("Saving latent vectors for the full dataset...")
        save_latent_vectors(model_ae, data_module.full_dataset, DEVICE,
                            output_path=str(analysis_output_dir / "all_latent_vectors.pt"),
                            batch_size=args.batch_size)  # Use current batch_size for this loader

        print("Running latent space optimization and geometry export...")
        opt_car_idx = args.optimization_car_index
        if opt_car_idx >= len(data_module.full_dataset):
            print(f"Warning: Requested car_index {opt_car_idx} for optimization is out of bounds. Using index 0.")
            opt_car_idx = 0

        if len(data_module.full_dataset) > 0:
            opt_results = run_latent_optimization(
                model_ae, model_cd, data_module.full_dataset, DEVICE,
                target_cd=args.optimization_target_cd, car_index=opt_car_idx,
                num_steps=args.optimization_num_steps, lr=args.optimization_lr,
                output_dir=analysis_output_dir / "optimized_geometry_output"
            )
            if opt_results and opt_results[1] is not None:  # Check if optimization produced output
                _, x_optimized_unnorm, optimized_edge_index, optimized_filepath = opt_results
                export_point_cloud_and_mesh(
                    x_optimized_unnorm, optimized_edge_index, optimized_filepath,
                    output_dir=analysis_output_dir / "optimized_geometry_output", suffix="optimized_final"
                )
            else:
                print(
                    "Skipping geometry export as optimization did not complete successfully or car_index was invalid.")
        else:
            print("Skipping latent space optimization as full dataset is empty.")

    else:
        print("Skipping latent vector saving and optimization as full dataset is empty or unavailable.")
    print("Analysis finished.")


def main():
    parser = argparse.ArgumentParser(description="Graph AE/CD Training and Analysis")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "analyze"],
                        help="Run mode: 'train' or 'analyze'")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint file for analysis mode or if resuming training.")
    parser.add_argument("--resume_checkpoint_path", type=str, default=None,
                        help="Path to checkpoint file to resume training from.")

    # Common hyperparameters for both modes (some might be primarily for training)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoaders")
    parser.add_argument("--num_dataloader_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--debug_file_limit", type=int, default=None,
                        help="Limit number of files for debugging (e.g., 500)")
    parser.add_argument("--experiment_name", type=str, default="default_run",
                        help="Name for the experiment run directory")

    # Training specific hyperparameters
    parser.add_argument("--in_channels", type=int, default=10, help="Input channels for AE")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Hidden channels for AE")
    parser.add_argument("--latent_dim", type=int, default=512, help="Latent dimension for AE")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--max_epochs", type=int, default=200, help="Max number of training epochs")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPU devices to use (e.g., 1, or 2 for multi-GPU)")
    parser.add_argument("--strategy", type=str, default="auto",
                        help="Distributed training strategy (e.g., ddp, ddp_find_unused_parameters_true)")

    # Analysis specific hyperparameters (especially for optimization part)
    parser.add_argument("--optimization_car_index", type=int, default=0,
                        help="Index of the car in the full_dataset for optimization")
    parser.add_argument("--optimization_target_cd", type=float, default=0.25, help="Target Cd value for optimization")
    parser.add_argument("--optimization_num_steps", type=int, default=100, help="Number of optimization steps")
    parser.add_argument("--optimization_lr", type=float, default=1e-3,
                        help="Learning rate for latent space optimization")

    args = parser.parse_args()

    if args.mode == "train":
        if args.resume_checkpoint_path:  # If resuming, set checkpoint_path for trainer.fit
            args.checkpoint_path = args.resume_checkpoint_path
        run_training(args)
    elif args.mode == "analyze":
        if not args.checkpoint_path:
            parser.error("--checkpoint_path is required for analysis mode.")
        run_analysis(args)
    else:
        parser.error(f"Invalid mode: {args.mode}. Choose 'train' or 'analyze'.")


if __name__ == '__main__':
    main()

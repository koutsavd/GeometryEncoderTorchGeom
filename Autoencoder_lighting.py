import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
from scipy import sparse


class CarGraphDataset(Dataset):
    def __init__(self, path, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)  # ✅ properly call super
        self.path = Path(path)
        self.files_x = sorted([f for f in self.path.rglob("*.npz") if "_adj" not in f.name])
        self.files_adj = sorted([f for f in self.path.rglob("*.npz") if "_adj" in f.name])

    def len(self):
        return len(self.files_x)

    def get(self, idx):
        x_data = np.load(self.files_x[idx])
        x = torch.tensor(x_data['x'], dtype=torch.float32)

        a = sparse.load_npz(self.files_adj[idx]).tocoo()
        edge_index = torch.tensor(np.vstack((a.row, a.col)), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)


class GraphAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.encoder_gnn = GCNConv(in_channels, hidden_channels)
        self.encoder_lin = torch.nn.Linear(hidden_channels, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, in_channels)

    def encode(self, x, edge_index):
        x = self.encoder_gnn(x, edge_index)
        x = F.relu(x)
        z = self.encoder_lin(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, data):
        z_node = self.encode(data.x, data.edge_index)
        z_graph = global_mean_pool(z_node, data.batch)
        z_graph_repeated = z_graph[data.batch]
        x_hat = self.decode(z_graph_repeated)

        # adjacency reconstruction (inner product)
        a_hat = torch.sigmoid(torch.matmul(z_node, z_node.T))

        return z_node, z_graph, x_hat, a_hat


class LitGraphAutoEncoder(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, latent_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphAutoEncoder(in_channels, hidden_channels, latent_dim)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        z_node, z_graph, x_hat, a_hat = self(batch)
        loss_x = F.mse_loss(x_hat, batch.x)

        adj_true = torch.zeros_like(a_hat)
        adj_true[batch.edge_index[0], batch.edge_index[1]] = 1.0
        loss_a = F.binary_cross_entropy(a_hat, adj_true)

        loss = loss_x + loss_a
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# === Training script ===
if __name__ == "__main__":
    dataset = CarGraphDataset("/Users/koutsavd/PycharmProjects/Geometry_GNN/Graphs")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LitGraphAutoEncoder(in_channels=8, hidden_channels=32, latent_dim=64)
    trainer = pl.Trainer(max_epochs=100, log_every_n_steps=1)
    trainer.fit(model, loader)

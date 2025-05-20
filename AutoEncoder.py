import os
import numpy as np
import torch
from torch.nn import Linear, MSELoss, BCELoss
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from scipy import sparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# === Dataset Loader ===
class CarGraphDataset(Dataset):
    def __init__(self, root_dir, indices=None):
        self.root_dir = Path(root_dir)
        all_files = sorted([f for f in self.root_dir.rglob("*.npz") if "_adj" not in f.name])
        self.files = [all_files[i] for i in indices] if indices else all_files
        super().__init__()

    def len(self):
        return len(self.files)

    def get(self, idx):
        feat_file = self.files[idx]
        adj_file = feat_file.with_name(feat_file.name.replace(".npz", "_adj.npz"))
        npz_data = np.load(feat_file)

        x = torch.tensor(npz_data["x"], dtype=torch.float32)
        e = torch.tensor(npz_data["e"], dtype=torch.float32)

        a = sparse.load_npz(adj_file).tocoo()
        edge_index = torch.tensor(np.vstack((a.row, a.col)), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=e)

# === Model ===
class GraphAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.encoder_gcn = GCNConv(in_channels, hidden_channels)
        self.encoder_lin = Linear(hidden_channels, latent_dim)
        self.decoder_lin = Linear(latent_dim, in_channels)

    def encode(self, x, edge_index):
        z = self.encoder_gcn(x, edge_index)
        z = torch.relu(z)
        z = self.encoder_lin(z)
        return z

    def decode(self, z):
        x_hat = self.decoder_lin(z)
        a_hat = torch.sigmoid(torch.matmul(z, z.T))
        return x_hat, a_hat

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z)

# === Training loop ===
def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad()

        x_hat, a_hat = model(batch.x, batch.edge_index)
        a_real = torch.zeros_like(a_hat)
        a_real[batch.edge_index[0], batch.edge_index[1]] = 1.0

        loss_x = MSELoss()(x_hat, batch.x)
        loss_a = BCELoss()(a_hat.view(-1), a_real.view(-1))
        loss = loss_x + loss_a

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

# === Main execution ===
if __name__ == "__main__":
    path = "/Users/koutsavd/PycharmProjects/Geometry_GNN/Graphs"
    all_indices = list(range(len(list(Path(path).rglob("*[!_adj].npz")))))
    train_idx, valtest_idx = train_test_split(all_indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(valtest_idx, test_size=0.5, random_state=42)

    train_set = CarGraphDataset(path, indices=train_idx)
    val_set = CarGraphDataset(path, indices=val_idx)
    test_set = CarGraphDataset(path, indices=test_idx)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)
    test_loader = DataLoader(test_set, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphAutoEncoder(in_channels=8, hidden_channels=32, latent_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    writer = SummaryWriter(log_dir="runs/plain_graph_ae")
    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, 101):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")

    # === Test ===
    model.load_state_dict(torch.load("checkpoints/best_model.pt"))
    test_loss = run_epoch(model, test_loader, optimizer, device, train=False)
    print(f"Final Test Loss: {test_loss:.4f}")
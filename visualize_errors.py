import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    # Load the metrics CSV
    df = pd.read_csv("Checkpoints/lightning_logs/version_0/metrics.csv")

    # Only keep rows that actually contain loss values
    df = df.dropna(subset=["train_loss_epoch", "val_loss"], how="all")

    # Group by epoch and get the last available value for each metric
    grouped = df.groupby("epoch").agg({
        "train_loss_epoch": "last",
        "val_loss": "last"
    }).reset_index()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(grouped["epoch"], grouped["train_loss_epoch"], label="Train Loss")
    plt.plot(grouped["epoch"], grouped["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

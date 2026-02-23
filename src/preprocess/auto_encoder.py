"""Shared AutoEncoder for cross-domain dimension reduction."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from src.utils import ensure_dir


class SharedDataset(Dataset):
    def __init__(self, features: torch.Tensor):
        self.features = features

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return self.features[idx].float()


class SharedAutoEncoder(nn.Module):
    def __init__(self, feature_dim=384, hidden_dim=256, target_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, p=2, dim=1)
        x_recon = self.decoder(z)
        return x_recon, z


class ReconstructLoss(nn.Module):
    def forward(self, x_recon, x):
        cos_sim = F.cosine_similarity(x_recon, x, dim=1)
        return torch.mean(1 - cos_sim)


def _train_model(dataset, model, optimizer, loss_fn, device,
                 epochs=100, batch_size=128):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for features in dataloader:
            features = features.to(device)
            optimizer.zero_grad()
            x_recon, _ = model(features)
            loss = loss_fn(x_recon, features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(dataloader)
            print(f"  AE Epoch [{epoch+1}/{epochs}], Loss: {avg:.6f}")
    return model


def train_autoencoder(shared_features: torch.Tensor, pair_name: str,
                      cfg_ae: dict, device: torch.device):
    """
    Train or load a shared autoencoder.

    Parameters
    ----------
    shared_features : torch.Tensor
        All user+item embeddings concatenated, shape [N, feature_dim].
    pair_name : str
        E.g. "CDs_and_Vinyl_Kindle_Store".
    cfg_ae : dict
        Keys: feature_dim, hidden_dim, target_dim, epochs, batch_size,
              learning_rate, overwrite.
    device : torch.device

    Returns
    -------
    Trained SharedAutoEncoder model (on device).
    """
    feature_dim = cfg_ae.get("feature_dim", 384)
    hidden_dim = cfg_ae.get("hidden_dim", 256)
    target_dim = cfg_ae.get("target_dim", 128)
    epochs = cfg_ae.get("epochs", 100)
    batch_size = cfg_ae.get("batch_size", 128)
    lr = cfg_ae.get("learning_rate", 0.0002)
    overwrite = cfg_ae.get("overwrite", False)

    model_path = f"model/{pair_name}_autoencoder.pth"
    ensure_dir("model")

    model = SharedAutoEncoder(feature_dim, hidden_dim, target_dim).to(device)
    loss_fn = ReconstructLoss().to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    if os.path.exists(model_path) and not overwrite:
        print(f"Loading pre-trained autoencoder from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training autoencoder...")
        dataset = SharedDataset(shared_features.to(device))
        model = _train_model(dataset, model, optimizer, loss_fn, device,
                             epochs=epochs, batch_size=batch_size)
        torch.save(model.state_dict(), model_path)
        print(f"Autoencoder saved to {model_path}")

    model.eval()
    return model

"""Train and compare visual models for robotic action prediction.

Run with --help for data paths and execution options. Recorded measurements
are stored separately from new runs; no dataset or model weights are bundled.
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

BaseModel = nn.Module


@dataclass
class DatasetOutput:
    """Keep observations and targets separate during batching."""

    input: dict
    output: dict


class NpDictDataset(Dataset):
    """Apply field-specific transforms to aligned, zero-offset observations."""

    def __init__(
        self, array_dict, dep_vars, indep_vars, transform_lkp, target_offset=0
    ):
        if target_offset != 0:
            raise ValueError("Only the zero-offset prediction pipeline is supported.")
        required = set(dep_vars + indep_vars + ["terminals"])
        missing = required - array_dict.keys()
        if missing:
            raise ValueError(f"Missing data fields: {sorted(missing)}")
        lengths = {len(array_dict[key]) for key in required}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("All fields must contain the same nonzero sample count.")
        self.arrays = array_dict
        self.inputs, self.targets = indep_vars, dep_vars
        self.transforms = transform_lkp

    def __len__(self):
        return len(self.arrays["actions"])

    def __getitem__(self, index):
        return DatasetOutput(
            input={
                key: self.transforms[key](self.arrays[key][index])
                for key in self.inputs
            },
            output={
                key: self.transforms[key](self.arrays[key][index])
                for key in self.targets
            },
        )


def load_all_files(directory, pattern, keys):
    """Concatenate aligned arrays from matching HDF5 files in a split."""
    directory = Path(directory)
    filenames = [name for name in os.listdir(directory) if re.match(pattern, name)]
    if not filenames:
        raise FileNotFoundError(f"No matching HDF5 files in {directory}")
    parts = {key: [] for key in keys}
    for filename in tqdm(filenames, desc=directory.name):
        with h5py.File(directory / filename, "r") as handle:
            for key in keys:
                parts[key].append(handle[key][()])
    return {key: np.concatenate(values, axis=0) for key, values in parts.items()}


class ConvolutionLayersConfig:
    """Track spatial dimensions for the convolution stacks used below."""

    def __init__(self, input_dim, input_channels, layers):
        self.layers = layers
        size, channels = input_dim, input_channels
        self.sizes, self.channels = [], []
        scalar = lambda value: value[0] if isinstance(value, tuple) else value
        for layer in layers:
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                kernel = scalar(layer.kernel_size)
                stride = scalar(layer.stride or layer.kernel_size)
                padding, dilation = scalar(layer.padding), scalar(layer.dilation)
                size = (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
            if isinstance(layer, nn.Conv2d):
                channels = layer.out_channels
            self.sizes.append(size)
            self.channels.append(channels)

    def get_output_dims(self):
        return self.sizes

    def get_output_channels(self):
        return self.channels


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(*config.layers)
        self.reset()

    def reset(self, gain=1.0):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        return self.layers(images)


class MLP(nn.Module):
    """Preserve the baseline's linear hidden layers without added activations."""

    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        dimensions = [input_dim, *hidden_dims, output_dim]
        self.layers = nn.Sequential(
            *[nn.Linear(left, right) for left, right in zip(dimensions, dimensions[1:])]
        )
        self.reset()

    def reset(self, gain=1.0):
        for index, layer in enumerate(self.layers):
            nn.init.xavier_uniform_(
                layer.weight, gain=gain if index < len(self.layers) - 1 else 1.0
            )
            nn.init.zeros_(layer.bias)

    def forward(self, values):
        return self.layers(values)


class JointCNNEncoder(nn.Module):
    def __init__(self, cnn, dense):
        super().__init__()
        self.cnn, self.dense = cnn, dense

    def reset(self, gain=1.0):
        self.cnn.reset(gain=gain)
        self.dense.reset(gain=gain)

    def forward(self, images):
        return self.dense(self.cnn(images).flatten(start_dim=1))


class TrackerBalancedLoss:
    """The baseline uses a sum of task losses; the tuned loop uses half weights."""

    def __init__(self, loss_lkp):
        self.losses = loss_lkp

    def __call__(self, predictions, targets):
        return sum(
            loss(predictions[key], targets[key]) for key, loss in self.losses.items()
        )


def train_baseline(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    """Run the baseline locally with its existing half-precision objective."""
    torch.manual_seed(1)
    model.reset({}, {}, {})
    history = []
    for epoch in range(1, epochs + 1):
        values = []
        for training, loader in ((True, train_loader), (False, val_loader)):
            model.train(training)
            # Half-precision accumulation matches the historical baseline convention.
            total, count = torch.tensor(0.0).half(), torch.tensor(0.0).half()
            with torch.set_grad_enabled(training):
                for batch in tqdm(
                    loader, desc=f"Baseline {epoch} {'train' if training else 'val'}"
                ):
                    inputs = {key: value.cuda() for key, value in batch.input.items()}
                    targets = {key: value.cuda() for key, value in batch.output.items()}
                    if training:
                        optimizer.zero_grad()
                    with torch.autocast(device_type="cuda"):
                        predictions = model(**inputs)
                        if training:
                            loss = criterion(predictions, targets)
                    if not training:
                        loss = criterion(predictions, targets)
                    total += loss.detach().cpu()
                    count += 1
                    if training:
                        loss.backward()
                        optimizer.step()
            values.append(float(total / count))
        history.append((epoch, *values))
        print(f"Baseline epoch {epoch}: train={values[0]:.5f}, val={values[1]:.5f}")
    return history


def plot_recorded_results(root):
    """Regenerate figures from recorded measurements without retraining."""
    with (root / "results/training_history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    def draw(ax, model, title):
        selected = [row for row in rows if row["model"] == model]
        epochs = [int(row["epoch"]) for row in selected]
        for key, label, color in (
            ("train_loss", "Training", "#2563eb"),
            ("validation_loss", "Validation", "#d97706"),
        ):
            ax.plot(
                epochs, [float(row[key]) for row in selected], label=label, color=color
            )
        ax.set(title=title, xlabel="Epoch", ylabel="Recorded loss")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

    (root / "figures").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    draw(axes[0], "baseline", "Baseline (sum of action losses)")
    draw(axes[1], "end_to_end", "Tuned model (mean of action losses)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Recorded loss (log scale)")
    fig.savefig(root / "figures/supervised_training.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    draw(ax, "vae", "VAE reconstruction + 0.05 x KL divergence")
    fig.savefig(root / "figures/vae_training.png", dpi=180)
    plt.close(fig)


def _summ(x):
    x = np.asarray(x)
    x = x.reshape(-1, x.shape[-1]) if x.ndim > 1 else x.reshape(-1, 1)
    return pd.DataFrame(
        {
            "min": x.min(axis=0),
            "p1": np.percentile(x, 1, axis=0),
            "mean": x.mean(axis=0),
            "std": x.std(axis=0),
            "p99": np.percentile(x, 99, axis=0),
            "max": x.max(axis=0),
        }
    )


def _nan_inf_report(x, name):
    x = np.asarray(x)
    print(f"{name}: NaN={np.isnan(x).any()}, Inf={np.isinf(x).any()}")


def _print_img_info(imgs, name, n=200):
    arr = np.asarray(imgs)
    print(f"\n{name} raw array shape: {arr.shape}, dtype: {arr.dtype}")
    if arr.ndim == 4:
        idx = np.linspace(0, arr.shape[0] - 1, min(n, arr.shape[0])).astype(int)
        samp = arr[idx]
    else:
        samp = arr[None, ...]
    print(f"{name} sample shape: {samp.shape}")
    print(f"{name} pixel range: min={samp.min()}, max={samp.max()}")
    if arr.ndim == 4:
        hwc = {(im.shape[0], im.shape[1], im.shape[2]) for im in arr}
        print(f"{name} unique HWC sizes: {len(hwc)} (show up to 5) -> {list(hwc)[:5]}")
    if samp.ndim == 4 and samp.shape[-1] in (1, 3):
        flat = samp.reshape(-1, samp.shape[-1]).astype(np.float32)
        ch_mean = flat.mean(axis=0)
        ch_std = flat.std(axis=0)
        print(f"{name} per-channel mean: {ch_mean}, std: {ch_std}")


# Add image and state embeddings before predicting movement and gripper logits.
class Baseline1(BaseModel):

    def __init__(
        self, image_encoder: JointCNNEncoder, obs_encoder: MLP, dense: MLP
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.obs_encoder = obs_encoder
        self.dense = dense

    def forward(self, images, obs):
        _img_enc = self.image_encoder(images)
        _obs_enc = self.obs_encoder(obs)
        _dense_enc = self.dense(_img_enc + _obs_enc)
        pos = _dense_enc[:, 0:3]
        grp = _dense_enc[:, 3:]
        return {"pos": pos, "grp": grp}

    def reset(self, image_encoder_kwargs, obs_encoder_kwargs, dense_kwargs):
        self.image_encoder.reset(**image_encoder_kwargs)
        self.obs_encoder.reset(**obs_encoder_kwargs)
        self.dense.reset(**dense_kwargs)


# Concatenate camera channels and robot-state fields; keep the baseline in float16.
def collate_func(input_list: List[DatasetOutput]) -> DatasetOutput:
    pos = []
    _grp = []
    images = []
    obs = []
    for val in input_list:
        images.append(
            torch.concat([val.input["front_cam_ob"], val.input["mount_cam_ob"]], dim=0)[
                None, :
            ]
        )
        obs.append(
            torch.concat(
                [
                    val.input["ee_cartesian_pos_ob"],
                    val.input["ee_cartesian_vel_ob"],
                    val.input["joint_pos_ob"],
                ],
                dim=0,
            )[None, :]
        )
        pos.append(val.output["actions"][0:3][None, :])
        _grp.append(val.output["actions"][-1:][None])
    _grp = torch.concat(_grp, dim=0)
    grp = torch.zeros(_grp.shape[0], 3)
    grp[torch.arange(len(grp)), _grp.squeeze().long()] = 1
    images_t = torch.concat(images, dim=0).half()
    obs_t = torch.concat(obs, dim=0).half()
    pos_t = torch.concat(pos, dim=0).half()
    grp_t = grp.half()
    return DatasetOutput(
        input={"images": images_t, "obs": obs_t}, output={"pos": pos_t, "grp": grp_t}
    )


def evaluate_holistic(model, val_loader):
    model.eval()
    all_pos_preds, all_pos_targs = ([], [])
    all_grp_preds, all_grp_targs = ([], [])
    print("Evaluating model on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            if torch.cuda.is_available():
                imgs = batch.input["images"].cuda()
                obs = batch.input["obs"].cuda()
                t_pos = batch.output["pos"].cuda()
                t_grp = batch.output["grp"].cuda()
            else:
                imgs = batch.input["images"]
                obs = batch.input["obs"]
                t_pos = batch.output["pos"]
                t_grp = batch.output["grp"]
            out = model(images=imgs, obs=obs)
            all_pos_preds.append(out["pos"].float().cpu().numpy())
            all_pos_targs.append(t_pos.float().cpu().numpy())
            all_grp_preds.append(torch.argmax(out["grp"], dim=1).cpu().numpy())
            all_grp_targs.append(torch.argmax(t_grp, dim=1).cpu().numpy())
    pos_preds = np.concatenate(all_pos_preds)
    pos_targs = np.concatenate(all_pos_targs)
    grp_preds = np.concatenate(all_grp_preds)
    grp_targs = np.concatenate(all_grp_targs)
    mse = mean_squared_error(pos_targs, pos_preds)
    mae = mean_absolute_error(pos_targs, pos_preds)
    print("\n=== Positional Metrics (Regression) ===")
    print(f"MSE: {mse:.5f} (Captures large errors)")
    print(f"MAE: {mae:.5f} (Average deviation)")
    acc = accuracy_score(grp_targs, grp_preds)
    f1 = f1_score(grp_targs, grp_preds, average="weighted")
    print("\n=== Gripper Metrics (Classification) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(
        classification_report(
            grp_targs, grp_preds, target_names=["Open", "Stay", "Close"]
        )
    )


def analyze_failures(model, val_loader, val_dataset, save_figure):
    model.float()
    model.eval()
    all_p_preds, all_p_targs = ([], [])
    all_g_preds, all_g_targs = ([], [])
    device = next(model.parameters()).device
    print("Running failure analysis (Model converted to Float32)...")
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch.input["images"].to(device).float()
            obs = batch.input["obs"].to(device).float()
            out = model(images=imgs, obs=obs)
            all_p_preds.append(out["pos"].float().cpu().numpy())
            all_p_targs.append(batch.output["pos"].float().cpu().numpy())
            all_g_preds.append(torch.argmax(out["grp"], dim=1).cpu().numpy())
            all_g_targs.append(torch.argmax(batch.output["grp"], dim=1).cpu().numpy())
    p_preds = np.concatenate(all_p_preds)
    p_targs = np.concatenate(all_p_targs)
    g_preds = np.concatenate(all_g_preds)
    g_targs = np.concatenate(all_g_targs)
    pos_errs = np.linalg.norm(p_preds - p_targs, axis=1)
    z_vals = p_targs[:, 2]
    print("\n=== Pattern 1: Position Error vs Target Movement (Z-axis) ===")
    bins = np.linspace(z_vals.min(), z_vals.max(), 6)
    bin_idxs = np.digitize(z_vals, bins)
    for i in range(1, 6):
        mask = bin_idxs == i
        if np.sum(mask) > 0:
            avg_err = np.mean(pos_errs[mask])
            print(
                f"Movement Range {bins[i - 1]:.2f} - {bins[i]:.2f}: Mean Error = {avg_err:.4f} (Count: {np.sum(mask)})"
            )
    print("\n=== Pattern 2: Gripper Classification Confusion ===")
    cm = confusion_matrix(g_targs, g_preds)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    labels = ["Open", "Stay", "Close"]
    for i, label in enumerate(labels):
        print(f"Class '{label}' Accuracy: {class_acc[i] * 100:.2f}%")
    print("\n=== Visualizing Worst Failures ===")
    worst_idxs = np.argsort(pos_errs)[-3:][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, idx in enumerate(worst_idxs):
        sample = val_dataset[idx]
        img = sample.input["front_cam_ob"]
        img_np = img.permute(1, 2, 0).numpy() * 0.5 + 0.5
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].set_title(
            f"Err: {pos_errs[idx]:.3f}\nTarget move Z: {p_targs[idx][2]:.2f}"
        )
        axes[i].axis("off")
    save_figure()


# Build the same multimodal batch layout in float32 for the tuned pipeline.
def safe_float_collate_func(input_list: List[DatasetOutput]) -> DatasetOutput:
    pos = []
    _grp = []
    images = []
    obs = []
    for val in input_list:
        images.append(
            torch.concat([val.input["front_cam_ob"], val.input["mount_cam_ob"]], dim=0)[
                None, :
            ]
        )
        obs.append(
            torch.concat(
                [
                    val.input["ee_cartesian_pos_ob"],
                    val.input["ee_cartesian_vel_ob"],
                    val.input["joint_pos_ob"],
                ],
                dim=0,
            )[None, :]
        )
        pos.append(val.output["actions"][0:3][None, :])
        _grp.append(val.output["actions"][-1:][None])
    _grp = torch.concat(_grp, dim=0)
    grp = torch.zeros(_grp.shape[0], 3)
    grp[torch.arange(len(grp)), _grp.squeeze().long()] = 1
    return DatasetOutput(
        input={
            "images": torch.concat(images, dim=0).float(),
            "obs": torch.concat(obs, dim=0).float(),
        },
        output={"pos": torch.concat(pos, dim=0).float(), "grp": grp.float()},
    )


class CustomMLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = int(input_dim)
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, int(h_dim)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = int(h_dim)
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def reset(self, **kwargs):
        for m in self.model:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


# Retain the two head calls and their separate training-time Dropout masks.
class TunedModel(BaseModel):

    def __init__(self, image_encoder, obs_encoder, dense):
        super().__init__()
        self.image_encoder = image_encoder
        self.obs_encoder = obs_encoder
        self.dense = dense

    def forward(self, images, obs):
        _img_enc = self.image_encoder(images)
        _obs_enc = self.obs_encoder(obs)
        return {
            "pos": self.dense(_img_enc + _obs_enc)[:, 0:3],
            "grp": self.dense(_img_enc + _obs_enc)[:, 3:],
        }

    def reset(self, **kwargs):

        def _reset(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        self.apply(_reset)


# Encode two 224 x 224 views into a 64-dimensional distribution and reconstruct them.
class ConvVAE(nn.Module):

    def __init__(self, input_channels=2, latent_dim=64):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.flatten_dim = 256 * 14 * 14
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return (self.fc_mu(h), self.fc_logvar(h))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc_decode(z).view(z.size(0), 256, 14, 14)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return (self.decode(z), mu, logvar)


# Sum pixel errors per sample; beta controls the KL regularization strength.
def vae_loss_fn(recon_x, x, mu, logvar, beta=0.1):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return (recon_loss + beta * kld_loss, recon_loss, kld_loss)


class RawDataset(Dataset):

    def __init__(self, nps_dict):
        self.front = nps_dict["front_cam_ob"]
        self.mount = nps_dict["mount_cam_ob"]

    def __len__(self):
        return len(self.front)

    def __getitem__(self, idx):
        f = torch.from_numpy(self.front[idx])
        if f.ndim == 2:
            f = f.unsqueeze(0)
        else:
            f = f.permute(2, 0, 1)
        m = torch.from_numpy(self.mount[idx])
        if m.ndim == 2:
            m = m.unsqueeze(0)
        else:
            m = m.permute(2, 0, 1)
        return torch.concat([f, m], dim=0)


def vae_collate(batch):
    return torch.stack(batch).float() / 255.0


# Freeze the VAE encoder and train the state branch and action head.
class LatentBCModel(BaseModel):

    def __init__(
        self,
        vae_model,
        obs_dim=15,
        latent_dim=64,
        hidden_dims=[256, 256],
        output_dim=6,
        dropout=0.3,
    ):
        super().__init__()
        self.vae = vae_model
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        layers = []
        in_dim = int(obs_dim)
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, int(h_dim)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = int(h_dim)
        self.obs_encoder = nn.Sequential(*layers)
        combined_dim = latent_dim + int(hidden_dims[-1])
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, images, obs):
        with torch.no_grad():
            mu, _ = self.vae.encode(images)
        obs_feat = self.obs_encoder(obs)
        combined = torch.cat([mu, obs_feat], dim=1)
        out = self.head(combined)
        return {"pos": out[:, 0:3], "grp": out[:, 3:]}

    def reset(self, **kwargs):

        def _reset(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        self.obs_encoder.apply(_reset)
        self.head.apply(_reset)


def evaluate_on_test(model, loader, device, name="Model"):
    model.eval()
    criterion_p = nn.MSELoss()
    criterion_g = nn.CrossEntropyLoss()
    total_loss_accum = 0.0
    pos_mse_accum = 0.0
    grp_acc_accum = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            images = batch.input["images"].to(device).float()
            obs = batch.input["obs"].to(device).float()
            target_pos = batch.output["pos"].to(device).float()
            target_grp = batch.output["grp"].to(device)
            target_grp_idx = torch.argmax(target_grp, dim=1)
            out = model(images, obs)
            l_p = criterion_p(out["pos"], target_pos)
            l_g = criterion_g(out["grp"], target_grp_idx)
            loss = 0.5 * l_p + 0.5 * l_g
            pred_grp = torch.argmax(out["grp"], dim=1)
            acc = (pred_grp == target_grp_idx).float().mean()
            total_loss_accum += loss.item()
            pos_mse_accum += l_p.item()
            grp_acc_accum += acc.item()
            batches += 1
    return {
        "Name": name,
        "Total Loss": total_loss_accum / batches,
        "Pos MSE": pos_mse_accum / batches,
        "Grp Acc": grp_acc_accum / batches,
    }


def count_trainable_params(model):
    return sum((p.numel() for p in model.parameters() if p.requires_grad))


def count_total_params(model):
    return sum((p.numel() for p in model.parameters()))


def main():
    parser = argparse.ArgumentParser(
        description="Train visual robotic action-prediction models."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with train/, val/, and test/ HDF5 splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/latest"),
        help="Directory for newly generated analysis figures and metrics.",
    )
    parser.add_argument(
        "--plot-recorded",
        action="store_true",
        help="Regenerate figures from historical CSV files, without training.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.plot_recorded:
        plot_recorded_results(root)
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Full training retains the CUDA/half-precision baseline and requires a CUDA GPU."
        )
    for split in ("train", "val", "test"):
        if not (args.data_dir / split).is_dir():
            raise FileNotFoundError(f"Missing data split: {args.data_dir / split}")
    project_options = SimpleNamespace(debug=False)
    train_dh = SimpleNamespace(loc=str(args.data_dir / "train"))
    val_dh = SimpleNamespace(loc=str(args.data_dir / "val"))
    test_dh = SimpleNamespace(loc=str(args.data_dir / "test"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_number = 0

    def save_figure():
        nonlocal figure_number
        figure_number += 1
        plt.savefig(
            args.output_dir / f"analysis_{figure_number:02d}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close()

    _keys = [
        "actions",
        "front_cam_ob",
        "mount_cam_ob",
        "terminals",
        "ee_cartesian_pos_ob",
        "ee_cartesian_vel_ob",
        "joint_pos_ob",
    ]
    train_nps = load_all_files(train_dh.loc, "train_[0-9]+.h5", keys=_keys)
    val_nps = load_all_files(val_dh.loc, "val_[0-9]+.h5", keys=_keys)

    display = print
    keys = [
        "actions",
        "front_cam_ob",
        "mount_cam_ob",
        "ee_cartesian_pos_ob",
        "ee_cartesian_vel_ob",
        "joint_pos_ob",
        "terminals",
    ]
    for k in keys:
        if k not in train_nps:
            print(f"Missing key in train_nps: {k}")
    print("\n=== Numeric summary (TRAIN) ===")
    for k in ["actions", "ee_cartesian_pos_ob", "ee_cartesian_vel_ob", "joint_pos_ob"]:
        x = np.asarray(train_nps[k])
        print(f"\n{k}: shape={x.shape}, dtype={x.dtype}")
        _nan_inf_report(x, k)
        display(_summ(x.astype(np.float32)))
    a = np.asarray(train_nps["actions"], dtype=np.float32)
    outside = np.mean((a < -1.0) | (a > 1.0))
    print(f"\nActions outside [-1,1]: {outside * 100:.4f}%")
    t = np.asarray(train_nps["terminals"])
    print(
        f"\nterminals: shape={t.shape}, dtype={t.dtype}, unique values={np.unique(t)[:10]}"
    )
    print(f"terminal rate: {np.mean(t.astype(np.float32)) * 100:.4f}%")
    _print_img_info(train_nps["front_cam_ob"], "front_cam_ob", n=200)
    _print_img_info(train_nps["mount_cam_ob"], "mount_cam_ob", n=200)

    cnn2_config = ConvolutionLayersConfig(
        input_dim=224,
        input_channels=2,
        layers=[
            nn.Conv2d(
                in_channels=2,
                out_channels=8,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        ],
    )
    _cnn_output_dim = cnn2_config.get_output_dims()
    _cnn_output_channels = cnn2_config.get_output_channels()
    _mlp_input_dim = int(
        _cnn_output_dim[-1] * _cnn_output_dim[-1] * _cnn_output_channels[-1]
    )
    image_encoder = JointCNNEncoder(
        cnn=CNN(cnn2_config),
        dense=MLP(input_dim=_mlp_input_dim, hidden_dims=[256], output_dim=128),
    )
    obs_encoder = MLP(input_dim=15, hidden_dims=[256, 256], output_dim=128)
    dense = MLP(input_dim=128, hidden_dims=[64, 32], output_dim=6)
    torch.manual_seed(1)
    model = Baseline1(image_encoder=image_encoder, obs_encoder=obs_encoder, dense=dense)

    pos_criterion = nn.MSELoss(reduction="mean")
    grp_criterion = nn.CrossEntropyLoss(reduction="mean")

    torch.manual_seed(1)
    model.reset({}, {}, {})
    exp_kwargs = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "half_precision": True,
        "target_offset": 0,
    }
    if project_options.debug:
        exp_kwargs["batch_size"] = 4
    safe_transforms = {}
    IMG_SIZE = (224, 224)
    safe_img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            lambda x: x.float() / 255.0,
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    safe_transforms["front_cam_ob"] = safe_img_transform
    safe_transforms["mount_cam_ob"] = safe_img_transform
    for k in [
        "actions",
        "ee_cartesian_pos_ob",
        "ee_cartesian_vel_ob",
        "joint_pos_ob",
        "terminals",
    ]:
        safe_transforms[k] = lambda x: torch.as_tensor(x, dtype=torch.float32)
    if exp_kwargs["half_precision"]:
        train_dataset = NpDictDataset(
            array_dict=train_nps,
            transform_lkp=safe_transforms,
            dep_vars=["actions"],
            indep_vars=[
                "front_cam_ob",
                "mount_cam_ob",
                "ee_cartesian_pos_ob",
                "ee_cartesian_vel_ob",
                "joint_pos_ob",
            ],
            target_offset=exp_kwargs["target_offset"],
        )
        val_dataset = NpDictDataset(
            array_dict=val_nps,
            transform_lkp=safe_transforms,
            dep_vars=["actions"],
            indep_vars=[
                "front_cam_ob",
                "mount_cam_ob",
                "ee_cartesian_pos_ob",
                "ee_cartesian_vel_ob",
                "joint_pos_ob",
            ],
            target_offset=exp_kwargs["target_offset"],
        )
    else:
        pass
    print(f"Dataset Loaded. Size: {len(train_dataset)}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=exp_kwargs["batch_size"],
        shuffle=True,
        collate_fn=collate_func,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=exp_kwargs["batch_size"],
        shuffle=False,
        collate_fn=collate_func,
        num_workers=0,
    )
    if exp_kwargs["half_precision"]:
        model = model.half()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=exp_kwargs["learning_rate"], eps=0.0001
        )
    else:
        model = model.float()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=exp_kwargs["learning_rate"], eps=0.0001
        )
    __criterion = TrackerBalancedLoss(
        loss_lkp={"pos": pos_criterion, "grp": grp_criterion}
    )
    first_batch = next(train_dataloader.__iter__())
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()
            _ = model(
                images=first_batch.input["images"].cuda(),
                obs=first_batch.input["obs"].cuda(),
            )
        else:
            _ = model(images=first_batch.input["images"], obs=first_batch.input["obs"])
    print("Test Pass: OK")

    train_baseline(model, train_dataloader, val_dataloader, optimizer, __criterion)

    evaluate_holistic(model, val_dataloader)

    analyze_failures(model, val_dataloader, val_dataset, save_figure)

    train_transform_aug = transforms.Compose(
        [
            transforms.ToTensor(),
            lambda x: x.float(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            lambda x: x.float(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    tuned_transforms_train = {
        "front_cam_ob": train_transform_aug,
        "mount_cam_ob": train_transform_aug,
        "actions": lambda x: torch.as_tensor(x, dtype=torch.float32),
        "ee_cartesian_pos_ob": lambda x: torch.as_tensor(x, dtype=torch.float32),
        "ee_cartesian_vel_ob": lambda x: torch.as_tensor(x, dtype=torch.float32),
        "joint_pos_ob": lambda x: torch.as_tensor(x, dtype=torch.float32),
        "terminals": lambda x: torch.as_tensor(x, dtype=torch.float32),
    }
    tuned_transforms_val = tuned_transforms_train.copy()
    tuned_transforms_val["front_cam_ob"] = val_transform
    tuned_transforms_val["mount_cam_ob"] = val_transform
    cnn_config = ConvolutionLayersConfig(
        input_dim=224,
        input_channels=2,
        layers=[
            nn.Conv2d(2, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        ],
    )
    _out_dim = cnn_config.get_output_dims()[-1]
    _out_ch = cnn_config.get_output_channels()[-1]
    _mlp_in = int(_out_dim**2 * _out_ch)
    tuned_image_encoder = JointCNNEncoder(
        cnn=CNN(cnn_config),
        dense=CustomMLP(
            input_dim=_mlp_in, hidden_dims=[256], output_dim=128, dropout=0.3
        ),
    )
    tuned_obs_encoder = CustomMLP(
        input_dim=15, hidden_dims=[256, 256], output_dim=128, dropout=0.3
    )
    tuned_dense = CustomMLP(
        input_dim=128, hidden_dims=[64, 32], output_dim=6, dropout=0.3
    )
    torch.manual_seed(1)
    tuned_model = TunedModel(tuned_image_encoder, tuned_obs_encoder, tuned_dense)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    tuned_model.to(device)
    tuned_model.float()
    train_data_float = {k: v.astype(np.float32) for k, v in train_nps.items()}
    val_data_float = {k: v.astype(np.float32) for k, v in val_nps.items()}
    tuned_train_dataset = NpDictDataset(
        array_dict=train_data_float,
        transform_lkp=tuned_transforms_train,
        dep_vars=["actions"],
        indep_vars=[
            "front_cam_ob",
            "mount_cam_ob",
            "ee_cartesian_pos_ob",
            "ee_cartesian_vel_ob",
            "joint_pos_ob",
        ],
        target_offset=0,
    )
    tuned_val_dataset = NpDictDataset(
        array_dict=val_data_float,
        transform_lkp=tuned_transforms_val,
        dep_vars=["actions"],
        indep_vars=[
            "front_cam_ob",
            "mount_cam_ob",
            "ee_cartesian_pos_ob",
            "ee_cartesian_vel_ob",
            "joint_pos_ob",
        ],
        target_offset=0,
    )
    tuned_train_loader = DataLoader(
        tuned_train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=safe_float_collate_func,
        num_workers=0,
    )
    tuned_val_loader = DataLoader(
        tuned_val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=safe_float_collate_func,
        num_workers=0,
    )
    tuned_optimizer = torch.optim.Adam(
        tuned_model.parameters(), lr=0.001, weight_decay=0.0001
    )
    pos_criterion = nn.MSELoss()
    grp_criterion = nn.CrossEntropyLoss()
    epochs = 10
    print("Starting Manual Training Loop...")
    for epoch in range(epochs):
        tuned_model.train()
        train_loss_accum = 0.0
        pbar = tqdm(tuned_train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in pbar:
            tuned_optimizer.zero_grad()
            images = batch.input["images"].to(device).float()
            obs = batch.input["obs"].to(device).float()
            target_pos = batch.output["pos"].to(device).float()
            target_grp = batch.output["grp"].to(device)
            out = tuned_model(images, obs)
            loss_p = pos_criterion(out["pos"], target_pos)
            target_grp_idx = torch.argmax(target_grp, dim=1)
            loss_g = grp_criterion(out["grp"], target_grp_idx)
            loss = 0.5 * loss_p + 0.5 * loss_g
            loss.backward()
            tuned_optimizer.step()
            train_loss_accum += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_train_loss = train_loss_accum / len(tuned_train_loader)
        tuned_model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in tuned_val_loader:
                images = batch.input["images"].to(device).float()
                obs = batch.input["obs"].to(device).float()
                target_pos = batch.output["pos"].to(device).float()
                target_grp = batch.output["grp"].to(device)
                out = tuned_model(images, obs)
                loss_p = pos_criterion(out["pos"], target_pos)
                target_grp_idx = torch.argmax(target_grp, dim=1)
                loss_g = grp_criterion(out["grp"], target_grp_idx)
                loss = 0.5 * loss_p + 0.5 * loss_g
                val_loss_accum += loss.item()
        avg_val_loss = val_loss_accum / len(tuned_val_loader)
        print(
            f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}"
        )
    print("Training Complete.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    vae_model = ConvVAE(input_channels=2, latent_dim=64).to(device).float()
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.001)
    train_loader_vae = DataLoader(
        RawDataset(train_nps), batch_size=32, shuffle=True, collate_fn=vae_collate
    )
    val_loader_vae = DataLoader(
        RawDataset(val_nps), batch_size=32, shuffle=False, collate_fn=vae_collate
    )
    epochs = 15
    beta = 0.05
    vae_history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kld_loss": []}
    print("Starting VAE Training...")
    for epoch in range(epochs):
        vae_model.train()
        train_loss = 0.0
        recon_loss = 0.0
        for imgs in tqdm(train_loader_vae, desc=f"Epoch {epoch + 1}"):
            imgs = imgs.to(device)
            vae_optimizer.zero_grad()
            recon, mu, logvar = vae_model(imgs)
            loss, r, k = vae_loss_fn(recon, imgs, mu, logvar, beta)
            loss.backward()
            vae_optimizer.step()
            train_loss += loss.item()
            recon_loss += r.item()
        avg_train = train_loss / len(train_loader_vae)
        vae_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs in val_loader_vae:
                imgs = imgs.to(device)
                recon, mu, logvar = vae_model(imgs)
                loss, _, _ = vae_loss_fn(recon, imgs, mu, logvar, beta)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader_vae)
        vae_history["train_loss"].append(avg_train)
        vae_history["val_loss"].append(avg_val)
        vae_history["recon_loss"].append(recon_loss / len(train_loader_vae))
        print(f"Epoch {epoch + 1}: Train={avg_train:.2f}, Val={avg_val:.2f}")
    print("Done.")

    if len(vae_history["kld_loss"]) == 0:
        print("Repairing missing KLD history from existing logs...")
        beta = 0.05
        vae_history["kld_loss"] = [
            (t - r) / beta
            for t, r in zip(vae_history["train_loss"], vae_history["recon_loss"])
        ]
    epochs_range = range(1, len(vae_history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(epochs_range, vae_history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs_range, vae_history["val_loss"], "r--o", label="Val Loss")
    ax1.set_title("VAE Total Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs_range, vae_history["recon_loss"], "g-", label="Reconstruction Loss")
    ax2.plot(epochs_range, vae_history["kld_loss"], "m-", label="KL Divergence")
    ax2.set_title("Loss Breakdown (Recon vs KL)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Component Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure()

    vae_model.eval()
    loader_to_use = val_loader_vae
    with torch.no_grad():
        batch = next(iter(loader_to_use))
        if isinstance(batch, torch.Tensor):
            imgs = batch
        elif hasattr(batch, "input"):
            imgs = batch.input["images"]
        else:
            imgs = (
                batch["input"]["images"]
                if isinstance(batch, dict) and "input" in batch
                else batch
            )
        imgs = imgs.to(device).float()
        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        recon_imgs, mu, logvar = vae_model(imgs)
        imgs_np = imgs.cpu().numpy()
        recon_np = recon_imgs.cpu().numpy()
        n_samples = 4
        fig, axes = plt.subplots(4, n_samples, figsize=(12, 12))
        for i in range(n_samples):
            axes[0, i].imshow(imgs_np[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[0, i].set_title("Orig Front")
            axes[0, i].axis("off")
            axes[1, i].imshow(recon_np[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[1, i].set_title("Recon Front")
            axes[1, i].axis("off")
            axes[2, i].imshow(imgs_np[i, 1], cmap="gray", vmin=0, vmax=1)
            axes[2, i].set_title("Orig Mount")
            axes[2, i].axis("off")
            axes[3, i].imshow(recon_np[i, 1], cmap="gray", vmin=0, vmax=1)
            axes[3, i].set_title("Recon Mount")
            axes[3, i].axis("off")
        plt.suptitle("Latent Space Analysis: 2-Channel Reconstruction")
        plt.tight_layout()
        save_figure()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    latent_model = LatentBCModel(vae_model=vae_model).to(device).float()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, latent_model.parameters()),
        lr=0.001,
        weight_decay=0.0001,
    )
    pos_criterion = nn.MSELoss()
    grp_criterion = nn.CrossEntropyLoss()
    loader_train = tuned_train_loader
    loader_val = tuned_val_loader
    epochs = 10
    print("Starting Latent BC Training...")
    for epoch in range(epochs):
        latent_model.train()
        latent_model.vae.eval()
        train_loss = 0.0
        for batch in tqdm(loader_train, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            images = batch.input["images"].to(device).float()
            obs = batch.input["obs"].to(device).float()
            target_pos = batch.output["pos"].to(device).float()
            target_grp = batch.output["grp"].to(device)
            out = latent_model(images, obs)
            l_p = pos_criterion(out["pos"], target_pos)
            l_g = grp_criterion(out["grp"], torch.argmax(target_grp, dim=1))
            loss = 0.5 * l_p + 0.5 * l_g
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(loader_train)
        latent_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in loader_val:
                images = batch.input["images"].to(device).float()
                obs = batch.input["obs"].to(device).float()
                target_pos = batch.output["pos"].to(device).float()
                target_grp = batch.output["grp"].to(device)
                out = latent_model(images, obs)
                l_p = pos_criterion(out["pos"], target_pos)
                l_g = grp_criterion(out["grp"], torch.argmax(target_grp, dim=1))
                val_loss += (0.5 * l_p + 0.5 * l_g).item()
        avg_val = val_loss / len(loader_val)
        print(f"Epoch {epoch + 1}: Train={avg_train:.4f}, Val={avg_val:.4f}")
    print("Done.")

    print("Loading Test Set...")
    data_dir = test_dh.loc
    print(f"Searching in: {data_dir}")
    _keys = [
        "front_cam_ob",
        "mount_cam_ob",
        "ee_cartesian_pos_ob",
        "ee_cartesian_vel_ob",
        "joint_pos_ob",
        "actions",
        "terminals",
    ]
    try:
        test_nps = load_all_files(data_dir, "test_[0-9]+.h5", keys=_keys)
    except Exception as e:
        print(f"Error loading files: {e}")
        test_nps = {}
    if len(test_nps) == 0:
        raise RuntimeError(f"No test files found in {data_dir}")
    else:
        print(f"Loaded {len(test_nps['actions'])} samples.")
    test_data_float = {k: v.astype(np.float32) for k, v in test_nps.items()}
    test_dataset = NpDictDataset(
        array_dict=test_data_float,
        transform_lkp=tuned_transforms_val,
        dep_vars=["actions"],
        indep_vars=[
            "front_cam_ob",
            "mount_cam_ob",
            "ee_cartesian_pos_ob",
            "ee_cartesian_vel_ob",
            "joint_pos_ob",
        ],
        target_offset=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=safe_float_collate_func,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")
    supervised_results = evaluate_on_test(
        tuned_model, test_loader, device, name="End-to-End"
    )
    latent_results = evaluate_on_test(
        latent_model, test_loader, device, name="Frozen VAE"
    )
    print("\n" + "=" * 65)
    print(
        f"{'Model Name':<20} | {'Total Loss':<12} | {'Pos MSE':<12} | {'Grp Acc':<12}"
    )
    print("-" * 65)
    print(
        f"{supervised_results['Name']:<20} | {supervised_results['Total Loss']:.4f}       | {supervised_results['Pos MSE']:.4f}       | {supervised_results['Grp Acc']:.2%}"
    )
    print(
        f"{latent_results['Name']:<20} | {latent_results['Total Loss']:.4f}       | {latent_results['Pos MSE']:.4f}       | {latent_results['Grp Acc']:.2%}"
    )
    print("=" * 65 + "\n")

    supervised_trainable = count_trainable_params(tuned_model)
    supervised_total = count_total_params(tuned_model)
    latent_trainable = count_trainable_params(latent_model)
    latent_total = count_total_params(latent_model)
    reduction = (1 - latent_trainable / supervised_trainable) * 100
    print("=== Model parameter counts ===")
    print(f"{'Metric':<30} | {'End-to-End':<15} | {'Frozen VAE':<15}")
    print("-" * 65)
    print(f"{'Total Parameters':<30} | {supervised_total:<15,} | {latent_total:<15,}")
    print(
        f"{'Trainable Parameters':<30} | {supervised_trainable:<15,} | {latent_trainable:<15,}"
    )
    print("-" * 65)
    print(f"Reduction in Trainable Params: {reduction:.2f}%")

    with (args.output_dir / "test_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(supervised_results))
        writer.writeheader()
        writer.writerows([supervised_results, latent_results])


if __name__ == "__main__":
    main()

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    num_epochs: int = 20
    device: str = "cpu"
    seed: int = 0


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)

    x_batch = torch.stack(xs, dim=0)
    y_batch = torch.stack(ys, dim=0)

    meta_dict = {}
    keys = metas[0].keys()
    for key in keys:
        meta_dict[key] = [m[key] for m in metas]

    return x_batch, y_batch, meta_dict


def build_dataloaders(
    train_dataset,
    test_dataset,
    batch_size: int,
) -> Dict[str, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_with_meta,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_with_meta,
    )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
    }


def move_batch_to_device(batch, device: torch.device):
    x, y, meta = batch
    x = x.to(device)
    y = y.to(device)
    return x, y, meta


def compute_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == y).float().sum().item()
    total = y.numel()
    return correct / total


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0
    num_batches = 0

    for batch in loader:
        x, y, _ = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        out = model(x)
        logits = out["logits"]
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == y).float().sum().item()
        total_examples += batch_size
        num_batches += 1

    mean_loss = total_loss / total_examples
    mean_acc = total_correct / total_examples

    return {
        "loss": mean_loss,
        "acc": mean_acc,
        "num_batches": num_batches,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    for batch in loader:
        x, y, _ = move_batch_to_device(batch, device)

        out = model(x)
        logits = out["logits"]
        loss = criterion(logits, y)

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == y).float().sum().item()
        total_examples += batch_size

    mean_loss = total_loss / total_examples
    mean_acc = total_correct / total_examples

    return {
        "loss": mean_loss,
        "acc": mean_acc,
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_l2_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        total += p.detach().pow(2).sum().item()
    return total ** 0.5
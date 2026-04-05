from __future__ import annotations

import torch

from src.data.modular_addition import (
    ModularAdditionConfig,
    build_modular_addition_datasets,
)
from src.models.mlp import MLPConfig, ModularMLP
from src.train_utils import (
    TrainConfig,
    build_dataloaders,
    count_parameters,
    evaluate,
    set_global_seed,
    train_one_epoch,
)


def main() -> None:
    # ---------------------------------
    # Config
    # ---------------------------------
    data_cfg = ModularAdditionConfig(
        p=97,
        train_frac=0.3,
        seed=0,
        one_hot=True,
    )

    model_cfg = MLPConfig(
        input_dim=2 * data_cfg.p,
        hidden_dim=256,
        output_dim=data_cfg.p,
        activation="relu",
        use_bias=True,
    )

    train_cfg = TrainConfig(
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-2,
        num_epochs=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
    )

    # ---------------------------------
    # Reproducibility
    # ---------------------------------
    set_global_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)

    # ---------------------------------
    # Data
    # ---------------------------------
    data_obj = build_modular_addition_datasets(data_cfg)
    train_dataset = data_obj["train_dataset"]
    test_dataset = data_obj["test_dataset"]

    loaders = build_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=train_cfg.batch_size,
    )
    train_loader = loaders["train_loader"]
    test_loader = loaders["test_loader"]

    # ---------------------------------
    # Model
    # ---------------------------------
    model = ModularMLP(model_cfg).to(device)
    print(model)
    print(f"device = {device}")
    print(f"num_parameters = {count_parameters(model)}")
    print("-" * 80)

    # ---------------------------------
    # Optimizer / loss
    # ---------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # ---------------------------------
    # Initial evaluation
    # ---------------------------------
    init_train_metrics = evaluate(model, train_loader, criterion, device)
    init_test_metrics = evaluate(model, test_loader, criterion, device)

    print("Before training:")
    print(
        f"  train loss = {init_train_metrics['loss']:.4f}, "
        f"train acc = {init_train_metrics['acc']:.4f}"
    )
    print(
        f"  test  loss = {init_test_metrics['loss']:.4f}, "
        f"test  acc = {init_test_metrics['acc']:.4f}"
    )
    print("-" * 80)

    # ---------------------------------
    # Train
    # ---------------------------------
    for epoch in range(1, train_cfg.num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss = {train_metrics['loss']:.4f}, "
            f"train acc = {train_metrics['acc']:.4f} | "
            f"test loss = {test_metrics['loss']:.4f}, "
            f"test acc = {test_metrics['acc']:.4f}"
        )

    print("-" * 80)
    print("Training smoke test complete.")


if __name__ == "__main__":
    main()
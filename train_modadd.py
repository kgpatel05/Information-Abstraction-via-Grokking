from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from src.analysis.plotting import plot_both
from src.data.modular_addition import (
    ModularAdditionConfig,
    build_modular_addition_datasets,
)
from src.experiments.io_utils import (
    append_metrics_row,
    ensure_dir,
    save_checkpoint,
    save_json,
)
from src.models.mlp import MLPConfig, ModularMLP
from src.models.transformer import TransformerConfig, ModularTransformer
from src.train_utils import (
    TrainConfig,
    build_dataloaders,
    count_parameters,
    evaluate,
    parameter_l2_norm,
    set_global_seed,
    train_one_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # shared/data args
    parser.add_argument("--arch", type=str, default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--data-seed", type=int, default=0)

    # mlp args
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--activation", type=str, default="relu")

    # transformer args
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=2)

    # train args
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)

    return parser.parse_args()


def make_run_name(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.arch == "mlp":
        arch_part = f"mlp_h{args.hidden_dim}"
    else:
        arch_part = (
            f"tr_d{args.d_model}"
            f"_h{args.n_heads}"
            f"_mlp{args.d_mlp}"
            f"_L{args.n_layers}"
        )

    return (
        f"{timestamp}"
        f"_modadd"
        f"_{args.arch}"
        f"_p{args.p}"
        f"_trainfrac{args.train_frac}"
        f"_{arch_part}"
        f"_wd{args.weight_decay}"
        f"_seed{args.seed}"
    )


def build_model_and_data_config(args):
    if args.arch == "mlp":
        data_cfg = ModularAdditionConfig(
            p=args.p,
            train_frac=args.train_frac,
            seed=args.data_seed,
            one_hot=True,
        )
        model_cfg = MLPConfig(
            input_dim=2 * data_cfg.p,
            hidden_dim=args.hidden_dim,
            output_dim=data_cfg.p,
            activation=args.activation,
            use_bias=True,
        )
        model = ModularMLP(model_cfg)
        return data_cfg, model_cfg, model

    if args.arch == "transformer":
        data_cfg = ModularAdditionConfig(
            p=args.p,
            train_frac=args.train_frac,
            seed=args.data_seed,
            one_hot=False,
        )
        model_cfg = TransformerConfig(
            vocab_size=data_cfg.p + 1,
            seq_len=3,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_mlp=args.d_mlp,
            n_layers=args.n_layers,
            output_dim=data_cfg.p,
            use_bias=True,
        )
        model = ModularTransformer(model_cfg)
        return data_cfg, model_cfg, model

    raise ValueError(f"Unknown arch: {args.arch}")


def main() -> None:
    args = parse_args()

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
    )

    set_global_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)

    data_cfg, model_cfg, model = build_model_and_data_config(args)
    model = model.to(device)

    run_name = make_run_name(args)
    run_dir = ensure_dir(Path("runs") / run_name)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    plot_dir = ensure_dir(run_dir / "plots")
    config_dir = ensure_dir(run_dir / "configs")
    metrics_path = run_dir / "metrics.csv"

    save_json(vars(args), config_dir / "cli_args.json")
    save_json(asdict(data_cfg), config_dir / "data_config.json")

    # model_cfg may be one of two dataclass types
    if hasattr(model_cfg, "__dict__"):
        save_json(asdict(model_cfg), config_dir / "model_config.json")
    save_json(asdict(train_cfg), config_dir / "train_config.json")

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    print("=" * 100)
    print(f"Run directory   : {run_dir}")
    print(f"Architecture    : {args.arch}")
    print(f"Device          : {device}")
    print(f"Parameters      : {count_parameters(model)}")
    print(f"Train examples  : {len(train_dataset)}")
    print(f"Test examples   : {len(test_dataset)}")
    print("=" * 100)

    global_step = 0

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

        global_step += int(train_metrics["num_batches"])

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "param_l2_norm": parameter_l2_norm(model),
            "learning_rate": train_cfg.learning_rate,
            "weight_decay": train_cfg.weight_decay,
            "train_frac": data_cfg.train_frac,
            "seed": train_cfg.seed,
            "arch": args.arch,
        }

        if args.arch == "mlp":
            row["hidden_dim"] = args.hidden_dim
        else:
            row["d_model"] = args.d_model
            row["n_heads"] = args.n_heads
            row["d_mlp"] = args.d_mlp
            row["n_layers"] = args.n_layers

        append_metrics_row(metrics_path, row)

        print(
            f"Epoch {epoch:04d} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_acc={row['train_acc']:.4f} | "
            f"test_loss={row['test_loss']:.4f} | "
            f"test_acc={row['test_acc']:.4f} | "
            f"step={global_step}"
        )

        if epoch % args.checkpoint_every == 0 or epoch == train_cfg.num_epochs:
            save_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                extra={
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "arch": args.arch,
                },
            )

    plot_both(metrics_csv=metrics_path, plot_dir=plot_dir)

    print("=" * 100)
    print("Training complete.")
    print(f"Metrics CSV      : {metrics_path}")
    print(f"Plots directory  : {plot_dir}")
    print(f"Checkpoints dir  : {checkpoint_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
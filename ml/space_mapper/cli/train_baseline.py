from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from space_mapper.datasets import SegmentationDataset, read_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny CPU-friendly U-Net baseline on a manifest subset.")
    parser.add_argument("--manifest", required=True, help="CSV manifest from prepare_sample_manifest.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resize", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-samples", type=int, help="Optional cap for quick CPU experiments.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--loss", choices=["bce", "dice", "bce_dice"], default="bce",
                        help="Loss function. Use bce_dice for imbalanced masks.")
    parser.add_argument("--pos-weight", default=None,
                        help="Positive-class weight for BCE: a number, or 'auto' to estimate from the data.")
    return parser


def _select_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false. Use --device cpu.")
    return torch.device(requested)


def _limit(records, max_samples: int | None):
    return records[:max_samples] if max_samples is not None else records


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from space_mapper.models import TinyUNet, require_torch
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    torch = require_torch()
    device = _select_device(torch, args.device)

    train_records = _limit(read_manifest(args.manifest, split="train"), args.max_samples)
    if not train_records:
        train_records = _limit(read_manifest(args.manifest), args.max_samples)
    if not train_records:
        raise SystemExit("Manifest contains no records to train on.")

    val_records = _limit(read_manifest(args.manifest, split="val"), args.max_samples)
    train_dataset = SegmentationDataset(train_records, resize=args.resize)
    val_dataset = SegmentationDataset(val_records, resize=args.resize) if val_records else None
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = (
        torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if val_dataset
        else None
    )

    first_image, _ = train_dataset[0]
    model = TinyUNet(in_channels=int(first_image.shape[0]), out_channels=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    from space_mapper.losses import compute_pos_weight, make_criterion

    pos_weight = None
    if args.pos_weight is not None:
        if str(args.pos_weight).lower() == "auto":
            pos_weight = compute_pos_weight(train_records)
            print(f"auto pos_weight = {float(pos_weight):.1f}" if pos_weight is not None else "pos_weight: no positives found")
        else:
            pos_weight = torch.tensor(float(args.pos_weight), dtype=torch.float32)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    loss_fn = make_criterion(args.loss, pos_weight=pos_weight)
    print(f"loss = {args.loss}")

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * images.size(0)
        train_loss /= len(train_dataset)

        val_loss = None
        if val_loader is not None:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    total += float(loss_fn(model(images), masks).detach().cpu()) * images.size(0)
            val_loss = total / len(val_dataset)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss if val_loss is not None else float("nan")})
        print(f"epoch={epoch} train_loss={train_loss:.4f}" + (f" val_loss={val_loss:.4f}" if val_loss is not None else ""))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": int(first_image.shape[0]),
            "base_channels": args.base_channels,
            "resize": args.resize,
            "epochs": args.epochs,
            "history": history,
        },
        output,
    )
    print(f"Saved TinyUNet checkpoint to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path

import torch 
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm


def train_one_epoch(
    model,
    optimizer,
    train_loader,
    device,
    epoch,
    loss_fn,
    scheduler=None,
    max_norm=None
):
    model.train()
    device = device 
    running_loss = 0.0
    running_accuracy = 0
    running_total = 0

    pbar = tqdm(
        iterable=enumerate(train_loader),
        total=len(train_loader), 
        desc=f"Epoch {epoch} train", 
        ncols=120
    )

    optimizer.zero_grad()

    for _, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images).to(device)
        loss = loss_fn(outputs, labels)
        loss.backward()

        if max_norm is not None:
            clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        batch_size = images.size(0)
        running_loss += batch_size * loss.item()
        preds = torch.argmax(outputs, dim=1)
        running_accuracy += (labels == preds).sum().item()
        running_total += batch_size
    
    epoch_loss = running_loss / running_total
    epoch_accuracy = running_accuracy / running_total
    lr = optimizer.param_groups[0]["lr"]
    return {"loss": epoch_loss, "accuracy": epoch_accuracy, "lr": lr}


def validate(model, val_loader, device, loss_fn):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0
    running_total = 0

    pbar = tqdm(
        iterable=enumerate(val_loader),
        total=len(val_loader),
        desc=f"Validate", 
        ncols=120
    )

    for _, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images).to(device)
        loss = loss_fn(outputs, labels)
        batch_size = images.size(0)
        running_loss += batch_size * loss.item()
        preds = torch.argmax(outputs, dim=1)
        running_accuracy += (labels == preds).sum().item()
        running_total += batch_size

    epoch_loss = running_loss / running_total
    epoch_accuracy = running_accuracy / running_total
    return {"loss": epoch_loss, "accuracy": epoch_accuracy}


def test(model, test_loader, device, loss_fn):
    return validate(model, test_loader, device, loss_fn)


def save_checkpoint(state, is_best, save_dir, epoch=None):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = "checkpoint_latest.pth"
    if epoch is not None:
        filename = f"epoch_{epoch:04d}.pth"
    path = Path(save_dir) / filename
    torch.save(state, path)
    if is_best:
        best_path = Path(save_dir) / "best.pth"
        torch.save(state, best_path)


def run_training(
    model,
    optimizer,
    train_loader,
    val_loader,
    device,
    workdir="experiments/exp",
    epochs=120,
    loss_fn=None,
    scheduler=None,
    max_norm=None,
    validate_every=5,
    save_every=20     
):
    best_val_acc = 0.0
    best_ckpt = None 
    workdir = Path(workdir)
    ckpt_dir = workdir / "checkpoints"
    metrics_log = []

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            loss_fn,
            scheduler=scheduler,
            max_norm=max_norm
        )

        val_metrics = None 
        if (epoch + 1) % validate_every == 0:
            val_metrics = validate(model, val_loader, device, loss_fn)

        is_best = False
        if val_metrics is not None:
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                is_best = True

        save_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc
        }

        if (epoch + 1) % save_every == 0 or is_best:
            save_checkpoint(save_state, str(ckpt_dir), epoch, is_best)
            if is_best:
                best_ckpt = str(ckpt_dir / "best.pth")

        log_entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        metrics_log.append(log_entry)
        print(f"Epoch {epoch} done. train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.2f}")
        if val_metrics:
            print(f"Val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.2f}")

    return {"best_ckpt": best_ckpt, "metrics_log": metrics_log}

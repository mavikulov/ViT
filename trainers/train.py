import yaml 
from pathlib import Path 

import torch 
import torch.nn as nn

from models.vit import ViT
from utils.download import load_cifar10
from utils.dataset import get_cifar_loaders
from trainers.engine import run_training


def main():
    config_path = Path("./configs")
    with open(config_path / "config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_cifar10(config['data']['root'])

    train_loader, val_loader, _ = get_cifar_loaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        batch_size=config['data']['batch_size']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = ViT(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    scheduler = None 
    loss_fn = nn.CrossEntropyLoss()

    run_training(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        workdir='experiments/exp_local',
        epochs=config['training'].get("epochs", 120),
        loss_fn=loss_fn,
        scheduler=scheduler,
        max_norm=1.0
    )


if __name__ == "__main__":
    main()
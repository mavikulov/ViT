from pathlib import Path
import yaml

import torch

from models.vit import ViT
from utils.dataset import get_cifar_loaders
from utils.download import load_cifar10
from trainers.engine import test


def main():
    config_path = Path("./configs")
    with open(config_path / "config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_val, X_test, y_train, y_val, y_test = load_cifar10()
    _, _, test_loader = get_cifar_loaders(X_train, y_train, X_val, y_val, X_test, y_test)

    model = ViT(**config["model"]).to(device)
    model = model.to(device)
    best_ckpt_path = Path("experiments/exp/checkpoints/best.pth")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = test(model, test_loader, device, loss_fn)
    print(f"Test loss: {metrics['loss']:.4f}, Test accuracy: {metrics['accuracy']:.2f}")

    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test[0:10]).to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        print("Predictions:", preds.cpu().numpy())


if __name__ == "__main__":
    main()

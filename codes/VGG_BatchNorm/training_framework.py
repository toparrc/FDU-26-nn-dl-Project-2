import os
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib as mpl
mpl.use("Agg")
import numpy as np
import torch
from tqdm import tqdm


DEVICE_ID = 0
DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")


def set_random_seeds(seed_value=0, device=DEVICE):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if str(device) != "cpu":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size
    return running_loss / total, correct / total


def train(model, optimizer, criterion, train_loader, val_loader, test_loader=None, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(DEVICE)
    losses_list = []
    grads = []
    history = []
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in tqdm(range(1, epochs_n + 1), unit="epoch"):
        if scheduler is not None:
            scheduler.step()
        model.train()
        loss_list = []
        grad = []
        train_loss = 0.0
        train_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            batch_size = y.size(0)
            loss_list.append(loss.item())
            grad.append(model.classifier[4].weight.grad.detach().norm().item())
            train_loss += loss.item() * batch_size
            train_correct += (prediction.argmax(dim=1) == y).sum().item()
            total += batch_size
            optimizer.step()

        train_loss /= total
        train_acc = train_correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            if best_model_path is not None:
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
        losses_list.append(loss_list)
        grads.append(grad)
        print(f"Epoch {epoch:03d}/{epochs_n:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} best_val_acc={best_val_acc:.4f}@{best_epoch}")

    test_metrics = {"best_epoch": best_epoch, "best_val_acc": best_val_acc}
    if test_loader is not None:
        if best_model_path is not None and os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        test_metrics.update({"test_loss": test_loss, "test_acc": test_acc})
        print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f}")

    return losses_list, grads, history, test_metrics


def flatten_losses(losses_list):
    return [loss for epoch_losses in losses_list for loss in epoch_losses]


def build_min_max_curves(loss_runs):
    flat_runs = [flatten_losses(losses) for losses in loss_runs]
    steps_n = min(len(run) for run in flat_runs)
    stacked = np.array([run[:steps_n] for run in flat_runs])
    return stacked.min(axis=0).tolist(), stacked.max(axis=0).tolist()

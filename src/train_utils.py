import random
import os
from pathlib import Path
import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def _set_mlflow_uri():
    import mlflow
    mlruns_path = Path(__file__).resolve().parent.parent / "mlruns"
    mlflow.set_tracking_uri(mlruns_path.as_uri())

SEED = 2026

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_true = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_true.extend(y.detach().cpu().tolist())
    loss = total_loss / len(all_true)
    acc = accuracy_score(all_true, all_preds)
    return loss, acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_true = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_true.extend(y.cpu().tolist())
    loss = total_loss / len(all_true)
    acc = accuracy_score(all_true, all_preds)
    return loss, acc, all_true, all_preds



def compute_metrics(y_true, y_pred, labels_order):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_order),
        "report": classification_report(y_true, y_pred, labels=labels_order, zero_division=0),
    }

def run_training(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience=None,
    min_delta=0.0,
    verbose=False,
    mlflow_params=None,
):
    def _run():
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_state, best_val = None, -1.0
        best_epoch = -1
        no_improve_count = 0

        for epoch in range(epochs):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)

            if mlflow_params is not None:
                import mlflow
                mlflow.log_metrics(
                    {"train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc},
                    step=epoch,
                )

            if verbose:
                print(
                    f"Epoch {epoch + 1:02d}/{epochs} | "
                    f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                    f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
                )

            if va_acc > (best_val + min_delta):
                best_val = va_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                no_improve_count = 0
            else:
                no_improve_count += 1

            if patience is not None and no_improve_count >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best epoch={best_epoch}, best val_acc={best_val:.4f})"
                    )
                break

        if mlflow_params is not None:
            import mlflow
            mlflow.log_metrics({"best_val_acc": best_val, "best_epoch": best_epoch})

        return history, best_state, best_epoch

    if mlflow_params is not None:
        import mlflow
        _set_mlflow_uri()
        with mlflow.start_run():
            mlflow.log_params(mlflow_params)
            return _run()
    return _run()



def save_artifact(path, payload):
    torch.save(payload, path)

def load_artifact(path, device, weights_only=False):
    return torch.load(path, map_location=device, weights_only=weights_only)
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm




def load_model(model_class, checkpoint_path, device):
    model = model_class().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def collect_probs(model, val_loader, device):
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)


def test_thresholds(all_probs, all_labels):
    thresholds = np.linspace(0.0, 1.0, 41)

    print("\nTHRESHOLD TESTING")
    print("Thresh | Precision | Recall | F1")
    print("-------------------------------------")

    for t in thresholds:
        preds = (all_probs >= t).astype(int)

        p = precision_score(all_labels, preds, zero_division=0)
        r = recall_score(all_labels, preds, zero_division=0)
        f = f1_score(all_labels, preds, zero_division=0)

        print(f"{t:5.3f} |   {p:6.3f}   | {r:6.3f} | {f:6.3f}")


def run_threshold_test(model_class, checkpoint_path, val_loader, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = load_model(model_class, checkpoint_path, device)

    print("Collecting probabilities...")
    all_probs, all_labels = collect_probs(model, val_loader, device)

    print("Testing thresholds...")
    test_thresholds(all_probs, all_labels)

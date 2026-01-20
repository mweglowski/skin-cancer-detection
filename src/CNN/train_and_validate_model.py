import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from sklearn.metrics import roc_auc_score
from custom_datasets import SkinLesionDataset, SkinLesionDatasetWithSynthetic
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.CNN.models.EfficientNetB3 import EfficientNetB3

THRESHOLD = 0.5

def custom_split(data_path):
    df = pd.read_csv(data_path)
    X = df
    y = df['target']
    groups = df["patient_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # splitter.split() returns tuple of indices
    for train_idx, test_idx in splitter.split(X, y, groups):
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        return df_train, df_test

def load_and_split_metadata(csv_path):
    og_df_path = r"/src/data/train-metadata.csv"
    og_df = pd.read_csv(og_df_path)
    add_df = pd.read_csv(csv_path)


    og_train_df, test_df = custom_split(r"/src/data/train-metadata.csv")
    add_df = add_df[add_df["patient_id"].notna() & (add_df["patient_id"] != "")]

    og_ids = set(og_df["isic_id"].dropna())
    add_df = add_df[~add_df["isic_id"].isin(og_ids)].reset_index(drop=True)


    test_patients = set(test_df["patient_id"].dropna())
    add_df = add_df[~add_df["patient_id"].isin(test_patients)].reset_index(drop=True)

    train_df_ben = og_train_df[og_train_df["target"] == 0].sample(n=3125, random_state = 42)
    train_df_mal = og_train_df[og_train_df["target"] == 1]
    train_df = pd.concat([train_df_ben, train_df_mal])

    add_df_mal = add_df[add_df["target"] == 1].sample(n = 2842)
    train_df = pd.concat([train_df, add_df_mal])

    val_ben = train_df[train_df["target"] == 0].sample(
        n=100, random_state=42
    )

    val_mal = train_df[train_df["target"] == 1].sample(
        n=100, random_state=42
    )

    val_df = pd.concat([val_ben, val_mal], ignore_index=True)

    train_df = train_df[~train_df["isic_id"].isin(val_df["isic_id"])]


    print(len(val_df))
    print(len(test_df))
    print(len(train_df))

    train_ids = set(train_df["isic_id"])
    test_ids = set(test_df["isic_id"])
    l1 = list(train_ids)[:10]
    l2 = list(test_ids)[:10]
    leak_ids = train_ids.intersection(test_ids)
    print("ISIC_ID leakage count:", len(leak_ids))
    if len(leak_ids) > 0: print("Leaked ISIC_IDs:", leak_ids)

    return train_df, val_df, test_df


def load_synthetic_df(synth_dir, label):
    synth_paths = list(Path(synth_dir).glob("*.png"))
    return pd.DataFrame({
        "isic_id": [p.name for p in synth_paths],
        "target": label
    })


def load_external_df(synth_dir, label):
    synth_paths = list(Path(synth_dir).glob("*.jpg"))
    return pd.DataFrame({
        "isic_id": [p.stem for p in synth_paths],
        "target": label
    })


def get_train_transforms(image_size=300):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=20,
            border_mode=0,
            p=0.7
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),

        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=10,
            val_shift_limit=5,
            p=0.3
        ),

        A.Resize(image_size, image_size),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])

def get_new_val_transforms(size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.RandomAffine(
            degrees=20,
            translate=(0.2, 0.2),
            shear=20,
            scale=(0.8, 1.2),
            fill=0
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])    ])

def get_val_test_transforms(size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size,size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def create_datasets(train_df, val_df, test_df, root_dir):
    val_transforms = test_transforms = get_val_test_transforms(300)
    new_train_transforms = get_new_val_transforms(300)
    train_transforms = get_train_transforms(300)

    train_ds = SkinLesionDatasetWithSynthetic(train_df, real_root=root_dir,
                                              synth_root="data/train-image/image", transforms=train_transforms)
    val_ds = SkinLesionDatasetWithSynthetic(val_df, real_root=root_dir,
                                              synth_root="data/train-image/image", transforms=val_transforms)
    test_ds = SkinLesionDataset(test_df, "../data/train-image/image", test_transforms)

    return train_ds, val_ds, test_ds


def create_loaders(train_ds, val_ds, test_ds, batch_size=32):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_acc = -1
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.view(-1, 1).float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        probs = torch.sigmoid(outputs).view(-1)
        preds = (probs > THRESHOLD).int()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        train_acc = 100 * correct / total
    return running_loss / len(loader), train_acc

def validate(model, loader, criterion, device, threshold=THRESHOLD):
    model.eval()
    running_loss = 0.0

    TP = FP = FN = TN = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels.view(-1, 1).float())
            running_loss += loss.item()

            probs = torch.sigmoid(logits).view(-1)
            preds = (probs > threshold).int()

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            TP += ((preds == 1) & (labels == 1)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()

    accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_loss = running_loss / len(loader)

    pauc = compute_partial_auc(np.array(all_probs), np.array(all_labels))
    confusion = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
    return avg_loss, accuracy, precision, recall, f1, pauc, confusion


def compute_partial_auc(y_prob, y_true):
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)

    v_gt = abs(y_true - 1)
    v_pred = 1.0 - y_prob

    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return partial_auc

def plot_training_curves(history, image_name):
    sns.set_theme('paper')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(data=history['train_acc'], label='Train Acc', ax=ax[0])
    sns.lineplot(data=history['val_acc'], label='Val Acc', ax=ax[0])
    ax[0].set_title('Accuracy')

    sns.lineplot(data=history['train_loss'], label='Train Loss', ax=ax[1])
    sns.lineplot(data=history['val_loss'], label='Val Loss', ax=ax[1])
    ax[1].set_title('Loss')

    plt.savefig(f"../images/{image_name}")
    plt.show()


def plot_confusion_matrix_from_conf(conf, image_name=None):

    cm = np.array([
        [conf["TN"], conf["FP"]],
        [conf["FN"], conf["TP"]]
    ])

    group_names = ['TN', 'FP', 'FN', 'TP']
    group_counts = [f"{v:0.0f}" for v in cm.flatten()]
    group_percentages = [f"{v:.2%}" for v in cm.flatten() / np.sum(cm)]

    labels = [
        f"{name}\n{count}\n{percent}"
        for name, count, percent in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt='',
        cmap='Blues',
        cbar=False,
        xticklabels=['Benign (0)', 'Malignant (1)'],
        yticklabels=['Benign (0)', 'Malignant (1)']
    )

    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")

    if image_name is not None:
        plt.savefig(f"../images/{image_name}")

    plt.tight_layout()
    plt.show()


import numpy as np



def main():
    train_df, val_df, test_df = load_and_split_metadata("../data/isic-dicm-17k/metadata-targeted.csv")

    train_ds, val_ds, test_ds = create_datasets(
        train_df, val_df, test_df, Path("../data/processed_17k")
    )

    train_loader, val_loader, test_loader = create_loaders(train_ds, val_ds, test_ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB3().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "train_acc": []}

    best_val_pauc = -1.0
    best_val_recall = -1.0
    best_model_path = "models/trained/Efficient.pt"

    for epoch in range(12):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_pauc, _ = validate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["train_acc"].append(tr_acc)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss {tr_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"Acc {val_acc:.2f} | Rec {val_rec:.3f} | Prec {val_prec:.3f} | F1 {val_f1:.3f} | pAUC {val_pauc:.4f}"
        )

        if val_pauc > best_val_pauc:
            best_val_pauc = val_pauc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved new best model (pauc={best_val_pauc:.3f})")

    model.load_state_dict(torch.load(best_model_path, map_location=device))


    val_loss, val_acc, val_prec, val_rec, val_f1, val_pauc, _ = validate(model, val_loader, criterion, device)
    print(f"FINAL VAL: Acc {val_acc:.2f} | Rec {val_rec:.3f} | Prec {val_prec:.3f} | F1 {val_f1:.3f} | pAUC {val_pauc:.4f}")

    test_loss, test_acc, test_prec, test_rec, test_f1, test_pauc, conf = validate(model, test_loader, criterion, device, threshold=0.825)
    print(f"FINAL TEST: Acc {test_acc:.2f} | Rec {test_rec:.3f} | Prec {test_prec:.3f} | F1 {test_f1:.3f} | pAUC {test_pauc:.4f}")

    plot_training_curves(history, "final_paper_plot.jpg")
    plot_confusion_matrix_from_conf(conf, "Model4_cm.jpg")

if __name__ == "__main__":
    main()

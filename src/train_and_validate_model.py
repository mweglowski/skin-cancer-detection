import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from custom_datasets import SkinLesionDataset, SkinLesionDatasetWithSynthetic
from src.models.EfficientNetB3 import EfficientNetB3
from src.models.UpgradedCNN import UpgradedCNN
from src.components.EarlyStopping import EarlyStopping
import albumentations as A
from albumentations.pytorch import ToTensorV2



def load_and_split_metadata(csv_path):
    df = pd.read_csv(csv_path)

    df_mal = df[df['target'] == 1]
    df_ben = df[df['target'] == 0]

    print(f"Total Malignant: {len(df_mal)}")
    print(f"Total Benign: {len(df_ben)}")

    test_mal, train_val_mal = train_test_split(df_mal, train_size=50)
    test_ben, train_val_ben = train_test_split(df_ben, train_size=1000)

    val_mal, train_mal = train_test_split(train_val_mal, train_size=200)
    val_ben, train_ben = train_test_split(train_val_ben, train_size=200)

    train_ben_down = train_ben.sample(n=2614)

    train_df = pd.concat([train_mal, train_ben_down]).sample(frac=1).reset_index(drop=True)
    val_df = pd.concat([val_mal, val_ben]).sample(frac=1).reset_index(drop=True)
    test_df = pd.concat([test_mal, test_ben]).sample(frac=1).reset_index(drop=True)

    print(f"Training Set: {len(train_df)} images ({train_df['target'].sum()} Malignant)")
    print(f"Val Set: {len(val_df)} images ({val_df['target'].sum()} Malignant)")
    print(f"Test Set: {len(test_df)} images ({test_df['target'].sum()} Malignant)")

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


def get_val_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((300, 300)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def create_datasets(train_df, val_df, test_df, root_dir):
    val_transforms = test_transforms = get_val_test_transforms()
    train_transforms = get_train_transforms(300)
    # train_ds = SkinLesionDataset(train_df, root_dir, train_transforms)
    # train_ds = SkinLesionDatasetWithSynthetic(train_df, real_root="data/train-image/image",
    #                                            synth_root="data/train-image/synthetic", transforms=train_transforms)
    train_ds = SkinLesionDatasetWithSynthetic(train_df, real_root="data/train-image/image",
                                              synth_root="data/malign_external_data", transforms=train_transforms)
    val_ds = SkinLesionDataset(val_df, root_dir, val_transforms)
    test_ds = SkinLesionDataset(test_df, root_dir, test_transforms)

    return train_ds, val_ds, test_ds


def create_loaders(train_ds, val_ds, test_ds, batch_size=32):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.view(-1, 1).float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).view(-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    TP = FP = FN = TN = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1).float())
            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).view(-1)

            TP += ((preds == 1) & (labels == 1)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()

    accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    avg_loss = running_loss / len(loader)

    return avg_loss, accuracy, precision, recall, f1


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


def plot_confusion_matrices(model, device, loaders, image_name):
    model.eval()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    with torch.no_grad():
        for ax, (name, loader) in zip(axes, loaders.items()):
            all_targets = []
            all_preds = []

            for images, labels in tqdm(loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float().view(-1)

                all_targets.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            # [[TN, FP], [FN, TP]]
            cm = confusion_matrix(all_targets, all_preds)

            group_names = ['TN', 'FP', 'FN', 'TP']
            group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]

            labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                      zip(group_names, group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(2, 2)

            sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax, cbar=False,
                        xticklabels=['Benign (0)', 'Malignant (1)'],
                        yticklabels=['Benign (0)', 'Malignant (1)'])

            ax.set_title(f'{name} Dataset')
            ax.set_ylabel('Actual Label')
            ax.set_xlabel('Predicted Label')
            plt.savefig(f'../images/{image_name}')

    plt.tight_layout()
    plt.show()


def main():

    train_df, val_df, test_df = load_and_split_metadata("data/train-metadata.csv")
    # synth_df = load_synthetic_df("data/train-image/synthetic", label=1)
    synth_df = load_external_df("data/malign_external_data", label=1)
    print(synth_df)
    train_df = pd.concat([train_df, synth_df], ignore_index=True)
    print(f"Real Training Set: {len(train_df)} images ({train_df['target'].sum()} Malignant)")

    train_ds, val_ds, test_ds = create_datasets(
        train_df, val_df, test_df, Path("data/train-image/image")
    )

    train_loader, val_loader, test_loader = create_loaders(train_ds, val_ds, test_ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB3().to(device)
    ##POS_Weight
    # pos_weight = torch.tensor([2.0], device=device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ##
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_f1 = -1.0
    best_val_recall = -1.0
    best_model_path = "models/trained/EfficientNetB3_1_to_1.pt"
   
    for epoch in range(20):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}: Train Acc {tr_acc:.2f} |  Val Acc {val_acc:.2f} | Val Rec {val_rec:.3f} | Val F1 {val_f1:.3f} | Val Prec {val_prec:.3f}")

        if val_rec > best_val_recall:
            best_val_recall = val_rec
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (recall = {best_val_recall:.3f})")

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    plot_training_curves(history, "external9_plot.jpg")

    val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
    print(
        f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | "
        f"Precision: {val_prec:.3f} | Recall: {val_rec:.3f} | F1: {val_f1:.3f}"
    )

    test_loss, test_acc, test_prec, test_rec, test_f1 = validate(model, test_loader, criterion, device)
    print(
        f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | "
        f"Precision: {test_prec:.3f} | Recall: {test_rec:.3f} | F1: {test_f1:.3f}"
    )

    loaders = {"Train": train_loader, "Validation": val_loader, "Test": test_loader}
    plot_confusion_matrices(model, device, loaders, "external9_mc.jpg")

if __name__ == "__main__":
    main()


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

from custom_datasets import SkinLesionDataset


def load_and_split_metadata(csv_path):
    df = pd.read_csv(csv_path)

    df_mal = df[df['target'] == 1]
    df_ben = df[df['target'] == 0]

    print(f"Total Malignant: {len(df_mal)}")
    print(f"Total Benign: {len(df_ben)}")

    test_mal, train_val_mal = train_test_split(df_mal, train_size=50)
    test_ben, train_val_ben = train_test_split(df_ben, train_size=1000)

    val_mal, train_mal = train_test_split(train_val_mal, train_size=50)
    val_ben, train_ben = train_test_split(train_val_ben, train_size=1000)

    train_ben_down = train_ben.sample(n=1465)

    train_df = pd.concat([train_mal, train_ben_down]).sample(frac=1).reset_index(drop=True)
    val_df = pd.concat([val_mal, val_ben]).sample(frac=1).reset_index(drop=True)
    test_df = pd.concat([test_mal, test_ben]).sample(frac=1).reset_index(drop=True)

    print(f"Training Set: {len(train_df)} images ({train_df['target'].sum()} Malignant)")
    print(f"Val Set: {len(val_df)} images ({val_df['target'].sum()} Malignant)")
    print(f"Test Set: {len(test_df)} images ({test_df['target'].sum()} Malignant)")

    return train_df, val_df, test_df


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def create_datasets(train_df, val_df, test_df, root_dir):
    t = get_transforms()

    train_ds = SkinLesionDataset(train_df, root_dir, t)
    val_ds = SkinLesionDataset(val_df, root_dir, t)
    test_ds = SkinLesionDataset(test_df, root_dir, t)

    return train_ds, val_ds, test_ds


def create_loaders(train_ds, val_ds, test_ds, batch_size=32):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * 64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.flatten(x)
        return self.fc(x)


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


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        patience: how many epochs to wait before stopping
        min_delta: minimum improvement in validation loss to count as progress
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True



def main():
    train_df, val_df, test_df = load_and_split_metadata("data/train-metadata.csv")

    train_ds, val_ds, test_ds = create_datasets(
        train_df, val_df, test_df, Path("data/train-image/image")
    )

    train_loader, val_loader, test_loader = create_loaders(train_ds, val_ds, test_ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    early_stopper = EarlyStopping(patience=6, min_delta=0.0001)

    for epoch in range(30):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print()
        print(f"Epoch {epoch+1}: Train Acc {tr_acc:.2f} | Val Acc {val_acc:.2f}")
        # early_stopper.step(val_loss)
        # if early_stopper.should_stop:
        #     print("Early stopping triggered — training halted.")
        #     break

    plot_training_curves(history, "balance_test_plots_1_to_5.jpg")

    print(
        f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | " f"Precision: {val_prec:.3f} | Recall: {val_rec:.3f} | F1: {val_f1:.3f}")

    test_loss, test_acc, test_prec, test_rec, test_f1 = validate(model, test_loader, criterion, device)

    print(
        f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | " f"Precision: {test_prec:.3f} | Recall: {test_rec:.3f} | F1: {test_f1:.3f}")

    loaders = {"Train": train_loader, "Validation": val_loader, "Test": test_loader}
    plot_confusion_matrices(model, device, loaders, "balance_test_cm_1_to_5.jpg")


if __name__ == "__main__":
    main()

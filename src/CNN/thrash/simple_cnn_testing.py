import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_datasets import SkinLesionDataset
from pathlib import Path
from tqdm import tqdm

df = pd.read_csv('../data/train-metadata.csv')

df_malignant = df[df['target'] == 1]
df_benign = df[df['target'] == 0]

print(f"Total Malignant: {len(df_malignant)}")
print(f"Total Benign: {len(df_benign)}")

test_mal, train_val_mal = train_test_split(df_malignant, test_size=None, train_size=50)
test_ben, train_val_ben = train_test_split(df_benign, test_size=None, train_size=1000)

val_mal, train_mal = train_test_split(train_val_mal, test_size=None, train_size=50)
val_ben, train_ben = train_test_split(train_val_ben, test_size=None, train_size=1000)

# 5. Create Training Set (The Balancing Act)
# We now have ~300 Malignant images left.
# We have ~398,000 Benign images left.
# WE CANNOT USE ALL BENIGN IMAGES. It will drown out the signal.

# Downsample Benign to a 1:5 ratio (300 Malignant : 1500 Benign)
# This gives the model a chance to actually see the cancer.
train_ben_downsampled = train_ben.sample(n=293)

# Concatenate back together
train_df = pd.concat([train_mal, train_ben_downsampled])
val_df = pd.concat([val_mal, val_ben])
test_df = pd.concat([test_mal, test_ben])

# Shuffle them
train_df = train_df.sample(frac=1).reset_index(drop=True)
val_df = val_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

print(f"Training Set: {len(train_df)} images ({train_df['target'].sum()} Malignant)")
print(f"Val Set: {len(val_df)} images ({val_df['target'].sum()} Malignant)")
print(f"Test Set: {len(test_df)} images ({test_df['target'].sum()} Malignant)")

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    # other augmentations for train dataset
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet mean and std
                         std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_ds = SkinLesionDataset(dataframe=train_df,
                             root_dir=Path('../data/train-image/image'),
                             transforms=train_transforms)
val_ds = SkinLesionDataset(dataframe=val_df,
                           root_dir=Path('../data/train-image/image'),
                           transforms=val_transforms)
test_ds = SkinLesionDataset(dataframe=test_df,
                            root_dir=Path('../data/train-image/image'),
                            transforms=test_transforms)

train_loader = DataLoader(train_ds, batch_size=32)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=32, 
                               kernel_size=3, 
                               padding=1) # 32, 128, 128
        self.batchNorm1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU(inplace=True) # inplace saves gpu memory (vram), modifies input tensor directly in memory rather than creating a new tensor for the output
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 32, 64, 64

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * 64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        return self.fc(x)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
model = SimpleCNN().to(device)
print(next(model.parameters()).device)
# maybe add pos_weight to tell model to pay more attention to malignant cases
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images) 
        
        loss = criterion(outputs, labels.view(-1, 1).float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        predicted = torch.sigmoid(outputs) > 0.5
        total += labels.size(0)
        correct += (predicted.view(-1) == labels).sum().item()
        
    avg_loss = running_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1).float())
            
            running_loss += loss.item()
            predicted = torch.sigmoid(outputs) > 0.5
            total += labels.size(0)
            correct += (predicted.view(-1) == labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

EPOCHS = 10

print("Starting Training...")

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print("-" * 30)
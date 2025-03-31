import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import cv2
import glob
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Paths from read file code
working_dir = '/kaggle/working/'
dataset_dir = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/'
TRAIN_LABELS_PATH = os.path.join(dataset_dir, 'train_labels.csv')
TRAIN_DIR = os.path.join(dataset_dir, 'train')
TEST_DIR = os.path.join(dataset_dir, 'test')
OUTPUT_MODEL = os.path.join(working_dir, 'fast_flagellum_model.pth')

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 48  # Fixed patch size for all dimensions
BATCH_SIZE = 32
EPOCHS = 10
NUM_WORKERS = 8
SEED = 42
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Simplified Model Architecture - only coordinate regression
class FastFlagellumDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv_layers(x)
        coords = self.regressor(features)
        return coords

# Optimized Data Loading with fixed dimensions
def load_tomogram(tomo_id, base_path):
    slice_paths = sorted(glob.glob(os.path.join(base_path, tomo_id, '*.jpg')))
    slices = []
    for i in range(PATCH_SIZE):
        if i < len(slice_paths):
            img = cv2.imread(slice_paths[i], 0)
            if img is not None:
                img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE))
                slices.append(img)
            else:
                slices.append(np.zeros((PATCH_SIZE, PATCH_SIZE)))
        else:
            slices.append(np.zeros((PATCH_SIZE, PATCH_SIZE)))

    volume = np.stack(slices)  # Shape: (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
    return volume

class TomogramDataset(Dataset):
    def __init__(self, ids, base_path, labels=None):
        self.ids = ids
        self.base_path = base_path
        self.labels = labels.set_index('tomo_id') if labels is not None else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tomo_id = self.ids[idx]
        try:
            vol = load_tomogram(tomo_id, self.base_path)
            vol = (vol - vol.mean()) / (vol.std() + 1e-6)  # Normalize
            vol_tensor = torch.tensor(vol, dtype=torch.float32).unsqueeze(0)  # Shape: (1, D, H, W)

            if self.labels is not None and tomo_id in self.labels.index:
                label = self.labels.loc[tomo_id]
                # Ensure we get a single row and convert to 1D array
                if isinstance(label, pd.Series):
                    coords = label[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values
                else:  # If we got a DataFrame (multiple rows)
                    coords = label[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].iloc[0].values
                coords = torch.tensor(coords, dtype=torch.float32) / PATCH_SIZE
                return vol_tensor, coords
            # Return dummy coords if no labels or tomo_id not found
            return vol_tensor, torch.zeros(3, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing {tomo_id}: {str(e)}")
            vol_tensor = torch.zeros((1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32)
            return vol_tensor, torch.zeros(3, dtype=torch.float32)

# Custom collate function
def custom_collate(batch):
    volumes = torch.stack([item[0] for item in batch])
    coords = torch.stack([item[1] for item in batch])
    return volumes, coords

# Loss Function - only coordinate regression
class FlagellumLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.coord_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        loss = self.coord_loss(outputs, targets)
        return loss

# Training Function
def train_fast(model, train_loader, criterion):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")

# Main Execution
if __name__ == '__main__':
    # Load data
    train_labels = pd.read_csv(TRAIN_LABELS_PATH).rename(columns={'id': 'tomo_id'})
    train_ids = sorted(os.listdir(TRAIN_DIR))

    # Create dataset and dataloader with custom collate
    dataset = TomogramDataset(train_ids, TRAIN_DIR, train_labels)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=NUM_WORKERS, pin_memory=True,
                            collate_fn=custom_collate)

    # Initialize model and loss
    model = FastFlagellumDetector().to(DEVICE)
    criterion = FlagellumLoss()

    # Train model
    train_fast(model, train_loader, criterion)

    # Save model
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"Model saved successfully to {OUTPUT_MODEL}")**

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import random
import os
from PIL import Image
import timm
import pandas as pd

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Path for the data set
DATA_DIR  = os.path.expanduser('~/Desktop/cse144_project/ucsc-cse-144-spring-2026-final-project')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR  = os.path.join(DATA_DIR, 'test')

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CorrectLabelDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        class_dirs = sorted(os.listdir(root), key=lambda x: int(x))
        for class_name in class_dirs:
            label = int(class_name)
            class_path = os.path.join(root, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Using all data for training for the final
full_dataset = CorrectLabelDataset(TRAIN_DIR, transform=train_transforms)
train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, num_workers=0)
print(f"Training on ALL {len(full_dataset)} images")

# Swin Base — pretrained on ImageNet-22k (22 million images!)
model = timm.create_model(
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
    pretrained=True,
    num_classes=100
)
model = model.to(device)
print("Swin Base (ImageNet-22k) loaded!")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def train_epoch():
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
    return total_loss / len(train_loader), correct / len(full_dataset) * 100

# Phase 1 
print("\n Phase 1: Head only (10 epochs)")
for name, p in model.named_parameters():
    if 'head' not in name:
        p.requires_grad = False
# learning rate a little higher
optimizer = torch.optim.Adam(
    [p for n, p in model.named_parameters() if 'head' in n],
    lr=1e-3
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    loss, acc = train_epoch()
    scheduler.step()
    print(f"Epoch {epoch+1}/10 | Loss: {loss:.3f} | Train Acc: {acc:.1f}%")

# Phase 2 — fine tuning all
# lower the lr
# also l2 reg for overfitting
print("\n Phase 2: Full fine-tune (30 epochs)")
for p in model.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_acc = 0
for epoch in range(30):
    loss, acc = train_epoch()
    scheduler.step()
    print(f"Epoch {epoch+1}/30 | Loss: {loss:.3f} | Train Acc: {acc:.1f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(DATA_DIR, 'pretrained.pth'))
        print(f"  Saved best model ({best_acc:.1f}%)")

print(f"\nTraining is complete! Best train acc: {best_acc:.1f}%")

# TTA
print("\n Running inference with TTA (5x) ")

tta_transforms = [
    val_transforms,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation((10, 10)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation((-10, -10)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
]

model.eval()
results = []
test_files = sorted(os.listdir(TEST_DIR), key=lambda x: int(x.replace('.jpg', '')))

with torch.no_grad():
    for i, fname in enumerate(test_files):
        img = Image.open(os.path.join(TEST_DIR, fname)).convert('RGB')
        probs = torch.zeros(100).to(device)
        for t in tta_transforms:
            inp = t(img).unsqueeze(0).to(device)
            probs += torch.softmax(model(inp).squeeze(), dim=0)
        pred = probs.argmax().item()
        results.append({'ID': fname, 'Label': pred})
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_files)} test images...")

df = pd.DataFrame(results)
out_path = os.path.join(DATA_DIR, 'submission_swin.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} predictions → {out_path}")
print(df.head(10))
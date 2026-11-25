

import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix

# Force real-time terminal output
def log(msg):
    print(msg, flush=True)

# ==========================================
# 1. SET PATHS (LOCAL ENVIRONMENT)
# ==========================================
log("[1/4] Setting up directory paths...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
drive_folder = os.path.join(BASE_DIR, 'Deepfake_Project')
final_dataset_root = os.path.join(BASE_DIR, 'dataset')

# ==========================================
# 2. LOCATE UNZIPPED FOLDERS & AGGREGATE
# ==========================================
log("\n[2/4] Scanning project folder for image files...")
os.makedirs(os.path.join(final_dataset_root, 'real'), exist_ok=True)
os.makedirs(os.path.join(final_dataset_root, 'fake'), exist_ok=True)

image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

if not os.path.exists(drive_folder):
    log(f" -> WARNING: Directory '{drive_folder}' not found!")
    log(f" -> Please make sure 'Deepfake_Project' folder is in: {BASE_DIR}")

copied_count = 0
for root, _, files in os.walk(drive_folder):
    folder_name = os.path.basename(root).lower()

    target_class = None
    if 'real' in folder_name:
        target_class = 'real'
    elif 'fake' in folder_name:
        target_class = 'fake'

    if target_class:
        log(f" -> Scanning: {root}")
        for file in files:
            if file.lower().endswith(image_extensions):
                src = os.path.join(root, file)
                unique_name = f"{os.path.basename(os.path.dirname(src))}_{file}"
                dst = os.path.join(final_dataset_root, target_class, unique_name)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied_count += 1
                    if copied_count % 100 == 0:
                        log(f"    Processed {copied_count} images...")

real_count = len(os.listdir(os.path.join(final_dataset_root, 'real')))
fake_count = len(os.listdir(os.path.join(final_dataset_root, 'fake')))
log(f"\nDataset Ready in Local Storage!")
log(f" -> REAL Images: {real_count}")
log(f" -> FAKE Images: {fake_count}")

if real_count == 0 and fake_count == 0:
    log("\n[ERROR] No images found! Check that 'Deepfake_Project' contains subfolders named 'real' and 'fake'.")
    exit()

# ==========================================
# 3. PYTORCH DATA PIPELINE & MODEL SETUP
# ==========================================
log("\n[3/4] Initializing PyTorch Pipeline & Model Architecture...")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=final_dataset_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f" -> Running on hardware device: {device}")

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class XceptionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.block1_rep_0_conv1 = SeparableConv2d(32, 64)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.block1_rep_0_conv1(x))
        x = torch.mean(x, dim=[2, 3])
        return self.fc(x)

model = XceptionDetector().to(device)

if os.path.exists(drive_folder):
    weight_files = [f for f in os.listdir(drive_folder) if f.endswith('.pth') or f.endswith('.pt')]
    if weight_files:
        weights_path = os.path.join(drive_folder, weight_files[0])
        log(f" -> Loading saved weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        log(" -> Model weights successfully loaded!")
    else:
        log(" -> Note: No .pth or .pt weight file found in Deepfake_Project.")

# ==========================================
# 4. EVALUATION LOOP & RESULTS PRINT
# ==========================================
log("\n[4/4] Executing Evaluation Loop...")
model.eval()
all_preds, all_labels = [], []

total_batches = len(dataloader)
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(dataloader, 1):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            log(f" -> Evaluated batch {batch_idx}/{total_batches}")

log("\n==========================================")
log("             EVALUATION RESULTS           ")
log("==========================================")
log("Confusion Matrix:")
log(str(confusion_matrix(all_labels, all_preds)))
log("\nClassification Report:")
log(classification_report(all_labels, all_preds, target_names=dataset.classes))
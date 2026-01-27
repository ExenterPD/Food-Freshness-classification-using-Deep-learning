# Combining EfficientNetV2-M (localized features) + MobileViT (global features)
# Optimized for local run with GPU acceleration

# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
import timm
import numpy as np
import os
from PIL import Image
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, log_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from collections import Counter
import xgboost as xgb

# Set up matplotlib for better plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Device Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple Silicon (MPS) detected!")
    print(f"   Device: MPS")
    use_mixed_precision = False
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ GPU detected!")
    use_mixed_precision = True
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
else:
    device = torch.device('cpu')
    print("⚠️ CPU detected.")
    use_mixed_precision = False

print(f" Using device: {device}")

# 🎯 Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Dataset Configuration
class FoodDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_training=True):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        try:
            import pillow_heif
            self.has_pillow_heif = True
            print("✅ pillow_heif is available for HEIC support.")
        except ImportError:
            raise ImportError("pillow_heif is required for HEIC image support. Install it with: pip install pillow_heif")
        self.folder_structure = self._get_folder_structure()
        self._load_dataset()

    def _get_folder_structure(self):
        folder_structure = {}
        main_categories = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        for main_category in main_categories:
            main_category_path = os.path.join(self.data_dir, main_category)
            if main_category.lower() in ['egg', 'paneer', 'dairy product']:
                folder_structure[main_category] = ['']
            else:
                subcategories = [d for d in os.listdir(main_category_path)
                                 if os.path.isdir(os.path.join(main_category_path, d))]
                folder_structure[main_category] = subcategories
        return folder_structure

    def _load_dataset(self):
        category_image_counts = Counter()
        for main_category, subcategories in self.folder_structure.items():
            for subcategory in subcategories:
                if main_category.lower() in ['egg', 'paneer', 'dairy product'] and subcategory == '':
                    category_path = os.path.join(self.data_dir, main_category)
                    category_key = main_category
                else:
                    category_path = os.path.join(self.data_dir, main_category, subcategory)
                    category_key = f"{main_category}_{subcategory}" if subcategory else main_category
                for quality in ['good', 'bad']:
                    quality_variations = [q for q in os.listdir(category_path)
                                          if os.path.isdir(os.path.join(category_path, q)) and q.lower() == quality]
                    if not quality_variations:
                        continue
                    quality_folder = quality_variations[0]
                    quality_path = os.path.join(category_path, quality_folder)
                    if os.path.exists(quality_path):
                        image_files = []
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.heic']:
                            image_files.extend([f for f in os.listdir(quality_path)
                                                if f.lower().endswith(ext.replace('*', '')) and not f.startswith('._')])
                        for img_file in image_files:
                            img_path = os.path.join(quality_path, img_file)
                            self.images.append(img_path)
                            self.labels.append(1 if quality == 'good' else 0)
                            class_name = f"{main_category}_{subcategory}_{quality_folder}" if subcategory else f"{main_category}_{quality_folder}"
                            self.class_names.append(class_name)
                            category_image_counts[category_key] += 1
        print(f"Loaded {len(self.images)} images")
        print(f"Good images: {sum(self.labels)}")
        print(f"Bad images: {len(self.labels) - sum(self.labels)}")
        print("\nImages per category:")
        for category, count in sorted(category_image_counts.items()):
            print(f"   {category}: {count} images")

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        try:
            if img_path.lower().endswith('.heic'):
                import pillow_heif
                heif_file = pillow_heif.read_heif(img_path)
                image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
                image = image.convert('RGB')
            else:
                image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_dir = "Master Dataset"
dataset = FoodDataset(data_dir, transform=train_transform, is_training=True)

# Split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

val_dataset.dataset.transform = val_transform
test_dataset.dataset.transform = val_transform

# Data loaders
batch_size = 32
num_workers = 0  
pin_memory = False  
persistent_workers = False

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Model Definitions
class EfficientNetV2FeatureExtractor(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        self.feature_dim = 1280
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output, features

class MobileViTFeatureExtractor(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model('mobilevit_s', pretrained=True)
        self.backbone.head.fc = nn.Identity()
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output, features

class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.efficientnet = EfficientNetV2FeatureExtractor(num_classes)
        self.mobilevit = MobileViTFeatureExtractor(num_classes)
        self.fusion_dim = self.efficientnet.feature_dim + self.mobilevit.feature_dim
        self.attention = nn.MultiheadAttention(self.fusion_dim, num_heads=8)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, features_only=False):
        eff_out, eff_feat = self.efficientnet(x)
        mvit_out, mvit_feat = self.mobilevit(x)
        fused_feat = torch.cat((eff_feat, mvit_feat), dim=1)
        attn_feat, _ = self.attention(fused_feat.unsqueeze(0), fused_feat.unsqueeze(0), fused_feat.unsqueeze(0))
        attn_feat = attn_feat.squeeze(0)
        output = self.fusion_layer(attn_feat)
        if features_only:
            return fused_feat
        return output

# Initialize model
model = HybridModel(num_classes=2).to(device)

# Optimizer and Scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def calculate_metrics(true_labels, predictions, probs=None):
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average='weighted'),
        'recall': recall_score(true_labels, predictions, average='weighted'),
        'f1': f1_score(true_labels, predictions, average='weighted'),
        'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0
    }
    if probs is not None:
        metrics['auroc'] = roc_auc_score(true_labels, probs[:, 1])
        metrics['loss'] = log_loss(true_labels, probs)
    return metrics

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_labels = []
    all_predictions = []
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} Training")):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Handle different device types
        if device.type == 'cuda' and use_mixed_precision:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # For MPS and CPU - no mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_predictions, None)
    return avg_loss, metrics

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    all_probs = np.array(all_probs)
    metrics = calculate_metrics(all_labels, all_predictions, all_probs)
    metrics['loss'] = avg_loss  # Override with CE loss for DL
    return avg_loss, metrics

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feat = model(images, features_only=True)
            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

# Training Configuration
class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
        self.training_times = []
        self.optimizers = []  # For boosting

    def add_epoch(self, train_loss, val_loss, train_metrics, val_metrics, epoch_time):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        self.training_times.append(epoch_time)

    def add_boosting_run(self, optimizer_name, train_metrics, val_metrics, test_metrics, train_time):
        self.optimizers.append(optimizer_name)
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        self.test_metrics.append(test_metrics)
        self.training_times.append(train_time)

    def get_best_epoch(self):
        val_f1_scores = [metrics['f1'] for metrics in self.val_metrics]
        return np.argmax(val_f1_scores)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*60}")
        print("DL Hybrid Results:")
        if self.train_losses:
            best_epoch = self.get_best_epoch()
            print(f"Best Epoch: {best_epoch + 1}")
            print(f"Best Validation F1 Score: {self.val_metrics[best_epoch]['f1']:.4f}")
            print(f"Best Validation Accuracy: {self.val_metrics[best_epoch]['accuracy']:.4f}")
            print(f"Total Training Time: {sum(self.training_times):.2f} seconds")
            print(f"Average Epoch Time: {np.mean(self.training_times):.2f} seconds")
        print("\nBoosting Results:")
        for i, opt in enumerate(self.optimizers):
            print(f"{opt} Results:")
            print(f"Train: {self.train_metrics[i]}")
            print(f"Val: {self.val_metrics[i]}")
            print(f"Test: {self.test_metrics[i]}")
            print(f"Training Time: {self.training_times[i]:.2f} seconds")
        print(f"{'='*60}")

history = TrainingHistory()

efficientnet_epochs = 2 if device.type == 'cuda' else 2
mobilevit_epochs = 2 if device.type == 'cuda' else 2
total_epochs = efficientnet_epochs + mobilevit_epochs

patience = 10
best_val_f1 = 0
patience_counter = 0

# Main Training Loop for Hybrid DL Model
print("Starting Deep Learning Training...")
start_time = time.time()
for epoch in range(total_epochs):
    epoch_start_time = time.time()
    if epoch < efficientnet_epochs:
        for param in model.mobilevit.parameters():
            param.requires_grad = False
        for param in model.efficientnet.parameters():
            param.requires_grad = True
        print(f"Epoch {epoch+1}: Training EfficientNet only")
    else:
        for param in model.efficientnet.parameters():
            param.requires_grad = False
        for param in model.mobilevit.parameters():
            param.requires_grad = True
        print(f"Epoch {epoch+1}: Training MobileViT only")
        
    train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
    val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
    scheduler.step()
    epoch_time = time.time() - epoch_start_time
    history.add_epoch(train_loss, val_loss, train_metrics, val_metrics, epoch_time)
    
    print(f"Epoch {epoch+1}/{total_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Val Metrics - Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
    
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        patience_counter = 0
        torch.save(model.state_dict(), 'best_hybrid_dl_model.pth')
        print(f"✅ New best model saved! Val F1: {best_val_f1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

total_training_time = time.time() - start_time
print(f"Total DL Training Time: {total_training_time:.2f} seconds")

# Final DL evaluation on test
print("\nEvaluating on Test Set...")
test_loss, test_metrics = evaluate_model(model, test_loader, criterion, device)
history.test_losses.append(test_loss)
history.test_metrics.append(test_metrics)

print(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

# Feature extraction
print("\nExtracting features for XGBoost...")
model.load_state_dict(torch.load('best_hybrid_dl_model.pth'))
X_train, y_train = extract_features(model, train_loader, device)
X_val, y_val = extract_features(model, val_loader, device)
X_test, y_test = extract_features(model, test_loader, device)

print(f"Feature shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Without optimization - Simple XGBoost
print("\n" + "="*60)
print("TRAINING XGBOOST WITHOUT OPTIMIZATION")
print("="*60)

print("\nTraining Default XGBoost (No Hyperparameter Tuning)...")
default_model = xgb.XGBClassifier(random_state=42)
start_time = time.time()
default_model.fit(X_train, y_train)
train_time = time.time() - start_time

# Training set predictions
y_pred_train = default_model.predict(X_train)
y_prob_train = default_model.predict_proba(X_train)
train_metrics_default = calculate_metrics(y_train, y_pred_train, y_prob_train)
train_metrics_default['time'] = train_time

# Validation set predictions
y_pred_val = default_model.predict(X_val)
y_prob_val = default_model.predict_proba(X_val)
val_metrics_default = calculate_metrics(y_val, y_pred_val, y_prob_val)

# Test set predictions
y_pred_test = default_model.predict(X_test)
y_prob_test = default_model.predict_proba(X_test)
test_metrics_default = calculate_metrics(y_test, y_pred_test, y_prob_test)

print("\n📊 DEFAULT XGBOOST RESULTS (No Optimization):")
print("Train Metrics:")
print(f"  Accuracy:  {train_metrics_default['accuracy']:.4f}")
print(f"  F1 Score:  {train_metrics_default['f1']:.4f}")
print(f"  Precision: {train_metrics_default['precision']:.4f}")
print(f"  Recall:    {train_metrics_default['recall']:.4f}")
print(f"  AUC-ROC:   {train_metrics_default['auroc']:.4f}")

print("\nValidation Metrics:")
print(f"  Accuracy:  {val_metrics_default['accuracy']:.4f}")
print(f"  F1 Score:  {val_metrics_default['f1']:.4f}")
print(f"  Precision: {val_metrics_default['precision']:.4f}")
print(f"  Recall:    {val_metrics_default['recall']:.4f}")
print(f"  AUC-ROC:   {val_metrics_default['auroc']:.4f}")

print("\nTest Metrics:")
print(f"  Accuracy:  {test_metrics_default['accuracy']:.4f}")
print(f"  F1 Score:  {test_metrics_default['f1']:.4f}")
print(f"  Precision: {test_metrics_default['precision']:.4f}")
print(f"  Recall:    {test_metrics_default['recall']:.4f}")
print(f"  AUC-ROC:   {test_metrics_default['auroc']:.4f}")

print(f"\n⏱️  Training Time: {train_time:.2f} seconds")

history.add_boosting_run('Default_XGBoost', train_metrics_default, val_metrics_default, test_metrics_default, train_time)

# Final Summary
print("\n" + "="*60)
print("FINAL SUMMARY - NO OPTIMIZATION USED")
print("="*60)
history.print_summary()

print("\n✅ Training completed successfully! Only default XGBoost (no optimization) was used.")
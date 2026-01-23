# Hybrid Model 2: DeiT (Small) + MobileViT with XGBoost + BOHB/Optuna Optimization
# With Real-time Epoch Visualization

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import numpy as np
import os
from PIL import Image
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from collections import Counter
import xgboost as xgb
import optuna
from hpbandster.core.worker import Worker
import ConfigSpace as CS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec

# Set up matplotlib for interactive plotting
plt.ion()  # Turn on interactive mode
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# Device Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple Silicon (MPS) detected!")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ GPU detected!")
else:
    device = torch.device('cpu')
    print("⚠️ CPU detected.")

print(f" Using device: {device}")

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Dataset Class
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
data_dir = "/Users/tanaypatel/Documents/projects/Machine_Learning/KFU/food_classification_project/Master Dataset"
dataset = FoodDataset(data_dir, transform=train_transform, is_training=True)

# Split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

val_dataset.dataset.transform = val_transform
test_dataset.dataset.transform = val_transform

# Data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Hybrid Model: DeiT + MobileViT
class HybridModel2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # DeiT backbone
        self.deit = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=0)
        self.deit_features = 384
        
        # MobileViT backbone
        self.mobilevit = timm.create_model('mobilevit_s', pretrained=True, num_classes=0)
        self.mobilevit_features = 640
        
        # Feature fusion
        self.fusion_dim = self.deit_features + self.mobilevit_features
        
        # Multi-head attention for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim, 
            num_heads=8,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, features_only=False):
        # Extract features from both backbones
        deit_features = self.deit(x)
        mvit_features = self.mobilevit(x)
        
        # Concatenate features
        fused_features = torch.cat((deit_features, mvit_features), dim=1)
        
        # Apply attention
        attended_features, _ = self.attention(
            fused_features.unsqueeze(1), 
            fused_features.unsqueeze(1), 
            fused_features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        if features_only:
            return fused_features  # Return raw fused features for boosting
        
        # Classification
        output = self.classifier(attended_features)
        return output

# Enhanced Training History with Real-time Visualization
class TrainingHistory:
    def __init__(self, model_name):
        self.model_name = model_name
        self.epoch_data = []
        self.fig = None
        self.axs = None
        self.setup_live_plot()
        
    def setup_live_plot(self):
        """Setup the live plotting figure"""
        self.fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=self.fig)
        
        self.axs = {
            'loss': self.fig.add_subplot(gs[0, 0]),
            'accuracy': self.fig.add_subplot(gs[0, 1]),
            'f1': self.fig.add_subplot(gs[0, 2]),
            'precision_recall': self.fig.add_subplot(gs[1, :]),
            'current_epoch': self.fig.add_subplot(gs[2, 0]),
            'metrics_table': self.fig.add_subplot(gs[2, 1:]),
        }
        
        self.fig.suptitle(f'{self.model_name} - Live Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
    def add_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, 
                  train_precision, val_precision, train_recall, val_recall, 
                  train_f1, val_f1, lr, epoch_time):
        self.epoch_data.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_precision': train_precision,
            'val_precision': val_precision,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'lr': lr,
            'time': epoch_time
        })
        
    def update_live_plot(self, current_epoch):
        """Update the live plot with current epoch data"""
        if not self.epoch_data:
            return
            
        epochs = [data['epoch'] for data in self.epoch_data]
        
        # Clear all axes
        for ax in self.axs.values():
            ax.clear()
        
        # 1. Loss plot
        train_losses = [data['train_loss'] for data in self.epoch_data]
        val_losses = [data['val_loss'] for data in self.epoch_data]
        self.axs['loss'].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
        self.axs['loss'].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s')
        self.axs['loss'].set_title('Training & Validation Loss')
        self.axs['loss'].set_xlabel('Epoch')
        self.axs['loss'].set_ylabel('Loss')
        self.axs['loss'].legend()
        self.axs['loss'].grid(True, alpha=0.3)
        
        # 2. Accuracy plot
        train_accs = [data['train_acc'] for data in self.epoch_data]
        val_accs = [data['val_acc'] for data in self.epoch_data]
        self.axs['accuracy'].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2, marker='o')
        self.axs['accuracy'].plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2, marker='s')
        self.axs['accuracy'].set_title('Training & Validation Accuracy')
        self.axs['accuracy'].set_xlabel('Epoch')
        self.axs['accuracy'].set_ylabel('Accuracy')
        self.axs['accuracy'].legend()
        self.axs['accuracy'].grid(True, alpha=0.3)
        
        # 3. F1 Score plot
        train_f1s = [data['train_f1'] for data in self.epoch_data]
        val_f1s = [data['val_f1'] for data in self.epoch_data]
        self.axs['f1'].plot(epochs, train_f1s, 'b-', label='Train F1', linewidth=2, marker='o')
        self.axs['f1'].plot(epochs, val_f1s, 'r-', label='Val F1', linewidth=2, marker='s')
        self.axs['f1'].set_title('Training & Validation F1 Score')
        self.axs['f1'].set_xlabel('Epoch')
        self.axs['f1'].set_ylabel('F1 Score')
        self.axs['f1'].legend()
        self.axs['f1'].grid(True, alpha=0.3)
        
        # 4. Precision/Recall plot
        train_prec = [data['train_precision'] for data in self.epoch_data]
        val_prec = [data['val_precision'] for data in self.epoch_data]
        train_rec = [data['train_recall'] for data in self.epoch_data]
        val_rec = [data['val_recall'] for data in self.epoch_data]
        
        self.axs['precision_recall'].plot(epochs, train_prec, 'g-', label='Train Precision', linewidth=2, marker='o')
        self.axs['precision_recall'].plot(epochs, val_prec, 'm-', label='Val Precision', linewidth=2, marker='s')
        self.axs['precision_recall'].plot(epochs, train_rec, 'c-', label='Train Recall', linewidth=2, marker='^')
        self.axs['precision_recall'].plot(epochs, val_rec, 'y-', label='Val Recall', linewidth=2, marker='d')
        self.axs['precision_recall'].set_title('Precision & Recall Over Time')
        self.axs['precision_recall'].set_xlabel('Epoch')
        self.axs['precision_recall'].set_ylabel('Score')
        self.axs['precision_recall'].legend()
        self.axs['precision_recall'].grid(True, alpha=0.3)
        
        # 5. Current Epoch Summary
        current_data = self.epoch_data[-1]
        metrics_text = (
            f"Epoch {current_epoch} Summary:\n\n"
            f"Train Loss: {current_data['train_loss']:.4f}\n"
            f"Val Loss: {current_data['val_loss']:.4f}\n"
            f"Train Acc: {current_data['train_acc']:.4f}\n"
            f"Val Acc: {current_data['val_acc']:.4f}\n"
            f"Train F1: {current_data['train_f1']:.4f}\n"
            f"Val F1: {current_data['val_f1']:.4f}\n"
            f"LR: {current_data['lr']:.2e}\n"
            f"Time: {current_data['time']:.1f}s"
        )
        self.axs['current_epoch'].text(0.1, 0.9, metrics_text, transform=self.axs['current_epoch'].transAxes, 
                                      fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        self.axs['current_epoch'].set_title('Current Epoch Metrics')
        self.axs['current_epoch'].axis('off')
        
        # 6. Metrics Table for all epochs
        if len(self.epoch_data) > 0:
            table_data = []
            columns = ['Epoch', 'TrLoss', 'VaLoss', 'TrAcc', 'VaAcc', 'TrF1', 'VaF1']
            for data in self.epoch_data[-8:]:  # Show last 8 epochs
                table_data.append([
                    data['epoch'],
                    f"{data['train_loss']:.3f}",
                    f"{data['val_loss']:.3f}",
                    f"{data['train_acc']:.3f}",
                    f"{data['val_acc']:.3f}",
                    f"{data['train_f1']:.3f}",
                    f"{data['val_f1']:.3f}"
                ])
            
            table = self.axs['metrics_table'].table(
                cellText=table_data,
                colLabels=columns,
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            self.axs['metrics_table'].set_title('Recent Epochs Summary')
            self.axs['metrics_table'].axis('off')
        
        plt.draw()
        plt.pause(0.1)
        
    def print_epoch_results(self, epoch):
        if epoch < len(self.epoch_data):
            data = self.epoch_data[epoch]
            print(f"Epoch {data['epoch']:2d} | "
                  f"Train Loss: {data['train_loss']:.4f} | Val Loss: {data['val_loss']:.4f} | "
                  f"Train Acc: {data['train_acc']:.4f} | Val Acc: {data['val_acc']:.4f} | "
                  f"Train Prec: {data['train_precision']:.4f} | Val Prec: {data['val_precision']:.4f} | "
                  f"Train Rec: {data['train_recall']:.4f} | Val Rec: {data['val_recall']:.4f} | "
                  f"Train F1: {data['train_f1']:.4f} | Val F1: {data['val_f1']:.4f} | "
                  f"LR: {data['lr']:.2e} | Time: {data['time']:.2f}s")
        
    def save_final_plot(self):
        """Save the final training plot"""
        plt.savefig(f'{self.model_name}_final_training_history.png', dpi=300, bbox_inches='tight')
        print(f"✅ Final training plot saved as: {self.model_name}_final_training_history.png")

# Metrics calculation
def calculate_metrics(true_labels, predictions, probs=None):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    if probs is not None and len(np.unique(true_labels)) > 1:
        try:
            metrics['auroc'] = roc_auc_score(true_labels, probs[:, 1])
        except:
            metrics['auroc'] = 0.0
            
    return metrics

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
    avg_loss = total_loss / len(dataloader)
    all_probs = np.array(all_probs)
    metrics = calculate_metrics(all_labels, all_predictions, all_probs)
    return avg_loss, metrics

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
    avg_loss = total_loss / len(dataloader)
    all_probs = np.array(all_probs)
    metrics = calculate_metrics(all_labels, all_predictions, all_probs)
    return avg_loss, metrics

# Feature extraction
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Extracting features")
        for images, lbls in pbar:
            images = images.to(device)
            feat = model(images, features_only=True)
            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

# Initialize model and history
model = HybridModel2(num_classes=2).to(device)
history = TrainingHistory("DeiT_MobileViT_Hybrid")

print("🚀 Starting Hybrid Model 2: DeiT + MobileViT")
print("=" * 80)

# Training configuration
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

total_epochs = 10
best_val_f1 = 0

# Main Training Loop with Live Visualization
for epoch in range(total_epochs):
    epoch_start = time.time()
    
    # Phase-based training
    if epoch < 5:
        # Train DeiT only
        for param in model.mobilevit.parameters():
            param.requires_grad = False
        for param in model.deit.parameters():
            param.requires_grad = True
        phase = "DeiT"
    else:
        # Train MobileViT only
        for param in model.deit.parameters():
            param.requires_grad = False
        for param in model.mobilevit.parameters():
            param.requires_grad = True
        phase = "MobileViT"
    
    print(f"\n📊 Epoch {epoch+1}/{total_epochs} - Training {phase} Phase")
    print("-" * 60)
    
    # Training
    train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
    
    # Validation
    val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
    
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    epoch_time = time.time() - epoch_start
    
    # Store epoch results
    history.add_epoch(
        epoch+1, train_loss, val_loss, 
        train_metrics['accuracy'], val_metrics['accuracy'],
        train_metrics['precision'], val_metrics['precision'],
        train_metrics['recall'], val_metrics['recall'],
        train_metrics['f1'], val_metrics['f1'],
        current_lr, epoch_time
    )
    
    # Print detailed epoch results
    history.print_epoch_results(epoch)
    
    # Update live plot
    history.update_live_plot(epoch+1)
    
    # Save best model
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        torch.save(model.state_dict(), 'best_hybrid_model2.pth')
        print(f"✅ New best model saved! Validation F1: {best_val_f1:.4f}")

print("\n" + "="*80)
print("🎯 Deep Learning Training Completed!")
print("="*80)

# Save final plot
history.save_final_plot()

# Turn off interactive mode
plt.ioff()

# Continue with feature extraction and optimization...
print("\n🔍 Loading best model and extracting features...")
model.load_state_dict(torch.load('best_hybrid_model2.pth'))
X_train, y_train = extract_features(model, train_loader, device)
X_val, y_val = extract_features(model, val_loader, device)
X_test, y_test = extract_features(model, test_loader, device)

print(f"Feature shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Save features
np.save('hybrid2_features_train.npy', X_train)
np.save('hybrid2_labels_train.npy', y_train)
np.save('hybrid2_features_val.npy', X_val)
np.save('hybrid2_labels_val.npy', y_val)
np.save('hybrid2_features_test.npy', X_test)
np.save('hybrid2_labels_test.npy', y_test)
print("✅ Features saved successfully!")

# Prepare data for XGBoost
X_combined = np.vstack([X_train, X_val])
y_combined = np.hstack([y_train, y_val])

# BOHB Optimization for XGBoost
print("\n🔧 Running BOHB Optimization for XGBoost...")

class XGBoostBOHBWorker(Worker):
    def __init__(self, X_train, y_train, X_val, y_val, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
    def compute(self, config, budget, **kwargs):
        model = xgb.XGBClassifier(
            n_estimators=int(budget),
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            reg_alpha=config['reg_alpha'],
            reg_lambda=config['reg_lambda'],
            random_state=42,
            n_jobs=1
        )
        model.fit(self.X_train, self.y_train)
        val_pred = model.predict(self.X_val)
        f1 = f1_score(self.y_val, val_pred, average='weighted')
        return {'loss': 1 - f1, 'info': {'f1_score': f1}}

try:
    # Create configuration space
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', 0.01, 0.3))
    config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_depth', 3, 10))
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter('subsample', 0.6, 1.0))
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter('colsample_bytree', 0.6, 1.0))
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter('reg_alpha', 0, 1))
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter('reg_lambda', 0, 1))
    
    # Create and run worker
    worker = XGBoostBOHBWorker(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        run_id='bohb_xgboost'
    )
    
    # Run BOHB optimization
    from hpbandster.optimizers import BOHB
    bohb = BOHB(
        configspace=config_space,
        run_id='bohb_xgboost',
        min_budget=50,
        max_budget=200,
        worker=worker
    )
    
    results = bohb.run(n_iterations=5)
    
    # Get best configuration
    best_config = results.get_incumbent_id()
    best_loss = results.get_incumbent_loss()
    
    print(f"🎯 BOHB Best Configuration: {best_config}")
    print(f"🎯 BOHB Best Loss: {best_loss:.4f}")
    
    # Train final model with best configuration
    best_xgb_bohb = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=best_config['max_depth'],
        learning_rate=best_config['learning_rate'],
        subsample=best_config['subsample'],
        colsample_bytree=best_config['colsample_bytree'],
        reg_alpha=best_config['reg_alpha'],
        reg_lambda=best_config['reg_lambda'],
        random_state=42
    )
    
    best_xgb_bohb.fit(X_combined, y_combined)
    
    # Evaluate BOHB model
    bohb_train_pred = best_xgb_bohb.predict(X_train)
    bohb_train_proba = best_xgb_bohb.predict_proba(X_train)
    bohb_test_pred = best_xgb_bohb.predict(X_test)
    bohb_test_proba = best_xgb_bohb.predict_proba(X_test)
    
    bohb_train_metrics = calculate_metrics(y_train, bohb_train_pred, bohb_train_proba)
    bohb_test_metrics = calculate_metrics(y_test, bohb_test_pred, bohb_test_proba)
    
    print("\n📊 BOHB-Optimized XGBoost Results:")
    print(f"Train - Acc: {bohb_train_metrics['accuracy']:.4f}, F1: {bohb_train_metrics['f1']:.4f}")
    print(f"Test  - Acc: {bohb_test_metrics['accuracy']:.4f}, F1: {bohb_test_metrics['f1']:.4f}")
    
except Exception as e:
    print(f"⚠️ BOHB optimization failed: {e}")
    print("Using simplified optimization for BOHB...")
    
    # Fallback optimization
    best_bohb_score = 0
    best_bohb_params = {}
    
    param_combinations = [
        {'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.8},
        {'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.9},
        {'learning_rate': 0.1, 'max_depth': 8, 'subsample': 0.7},
        {'learning_rate': 0.2, 'max_depth': 10, 'subsample': 0.6}
    ]
    
    for params in param_combinations:
        model = xgb.XGBClassifier(
            n_estimators=100,
            **params,
            random_state=42
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        score = f1_score(y_val, val_pred, average='weighted')
        
        if score > best_bohb_score:
            best_bohb_score = score
            best_bohb_params = params
    
    best_xgb_bohb = xgb.XGBClassifier(
        n_estimators=200,
        **best_bohb_params,
        random_state=42
    )
    best_xgb_bohb.fit(X_combined, y_combined)
    
    bohb_train_pred = best_xgb_bohb.predict(X_train)
    bohb_train_proba = best_xgb_bohb.predict_proba(X_train)
    bohb_test_pred = best_xgb_bohb.predict(X_test)
    bohb_test_proba = best_xgb_bohb.predict_proba(X_test)
    
    bohb_train_metrics = calculate_metrics(y_train, bohb_train_pred, bohb_train_proba)
    bohb_test_metrics = calculate_metrics(y_test, bohb_test_pred, bohb_test_proba)
    
    print("\n📊 Simplified BOHB XGBoost Results:")
    print(f"Train - Acc: {bohb_train_metrics['accuracy']:.4f}, F1: {bohb_train_metrics['f1']:.4f}")
    print(f"Test  - Acc: {bohb_test_metrics['accuracy']:.4f}, F1: {bohb_test_metrics['f1']:.4f}")

# Optuna Optimization for XGBoost
print("\n🔧 Running Optuna Optimization for XGBoost...")

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    return f1_score(y_val, val_pred, average='weighted')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print(f"🎯 Optuna Best Trial: {study.best_trial.value:.4f}")
print(f"🎯 Optuna Best Hyperparameters: {study.best_trial.params}")

# Train final Optuna model
best_xgb_optuna = xgb.XGBClassifier(**study.best_trial.params, random_state=42)
best_xgb_optuna.fit(X_combined, y_combined)

# Evaluate Optuna model
optuna_train_pred = best_xgb_optuna.predict(X_train)
optuna_train_proba = best_xgb_optuna.predict_proba(X_train)
optuna_test_pred = best_xgb_optuna.predict(X_test)
optuna_test_proba = best_xgb_optuna.predict_proba(X_test)

optuna_train_metrics = calculate_metrics(y_train, optuna_train_pred, optuna_train_proba)
optuna_test_metrics = calculate_metrics(y_test, optuna_test_pred, optuna_test_proba)

print("\n📊 Optuna-Optimized XGBoost Results:")
print(f"Train - Acc: {optuna_train_metrics['accuracy']:.4f}, F1: {optuna_train_metrics['f1']:.4f}")
print(f"Test  - Acc: {optuna_test_metrics['accuracy']:.4f}, F1: {optuna_test_metrics['f1']:.4f}")

# Final Summary
print("\n" + "="*80)
print("🎉 HYBRID MODEL 2 - FINAL SUMMARY")
print("="*80)
print("Architecture: DeiT + MobileViT")
print(f"Feature Dimension: {X_train.shape[1]}")
print(f"Total Training Samples: {len(X_train)}")
print(f"Total Test Samples: {len(X_test)}")

print("\n📊 Optimization Results Comparison:")
print("\nBOHB-Optimized XGBoost:")
print(f"  Train Accuracy: {bohb_train_metrics['accuracy']:.4f}")
print(f"  Train F1-Score: {bohb_train_metrics['f1']:.4f}")
print(f"  Test Accuracy:  {bohb_test_metrics['accuracy']:.4f}")
print(f"  Test F1-Score:  {bohb_test_metrics['f1']:.4f}")

print("\nOptuna-Optimized XGBoost:")
print(f"  Train Accuracy: {optuna_train_metrics['accuracy']:.4f}")
print(f"  Train F1-Score: {optuna_train_metrics['f1']:.4f}")
print(f"  Test Accuracy:  {optuna_test_metrics['accuracy']:.4f}")
print(f"  Test F1-Score:  {optuna_test_metrics['f1']:.4f}")

print(f"\n✅ Hybrid Model 2 training and optimization completed!")
print(f"📊 Training curves saved as: DeiT_MobileViT_Hybrid_final_training_history.png")
print(f"💾 Features saved as: hybrid2_features_*.npy")
print(f"💾 Models saved as: best_hybrid_model2.pth")

# Show final plot
plt.show()
# Hybrid Model 1: EfficientNetV2M + MobileViT for Food Quality Classification
# Complete implementation with detailed epoch logging and optimization

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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from collections import Counter
import xgboost as xgb
import optuna
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
import ConfigSpace as CS
import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Device Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple Silicon (MPS) detected!")
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Dataset Class for Updated Structure
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
        main_categories = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.')]
        for main_category in main_categories:
            main_category_path = os.path.join(self.data_dir, main_category)
            subcategories = [d for d in os.listdir(main_category_path)
                             if os.path.isdir(os.path.join(main_category_path, d)) and not d.startswith('.')]
            folder_structure[main_category] = subcategories
        return folder_structure

    def _load_dataset(self):
        category_image_counts = Counter()
        for main_category, subcategories in self.folder_structure.items():
            for subcategory in subcategories:
                category_path = os.path.join(self.data_dir, main_category, subcategory)
                category_key = f"{main_category}_{subcategory}" if subcategory else main_category
                for quality in ['Good', 'Bad']:
                    quality_variations = [q for q in os.listdir(category_path)
                                          if os.path.isdir(os.path.join(category_path, q)) and not q.startswith('.') and q.lower() == quality.lower()]
                    if not quality_variations:
                        continue
                    quality_folder = quality_variations[0]
                    quality_path = os.path.join(category_path, quality_folder)
                    if os.path.exists(quality_path):
                        image_files = []
                        for ext in ['.jpg', '.jpeg', '.png', '.heic']:
                            image_files.extend([f for f in os.listdir(quality_path)
                                                if f.lower().endswith(ext) and not f.startswith('._') and os.path.isfile(os.path.join(quality_path, f))])
                        for img_file in image_files:
                            img_path = os.path.join(quality_path, img_file)
                            self.images.append(img_path)
                            self.labels.append(1 if quality == 'Good' else 0)
                            class_name = f"{main_category}_{subcategory}_{quality_folder}"
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

class HybridModel1(nn.Module):
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
        
        # Apply attention
        attn_feat, _ = self.attention(fused_feat.unsqueeze(0), fused_feat.unsqueeze(0), fused_feat.unsqueeze(0))
        attn_feat = attn_feat.squeeze(0)
        
        output = self.fusion_layer(attn_feat)
        if features_only:
            return fused_feat  # Return raw fused features for boosting
        return output

# Enhanced Training History with Detailed Epoch Logging
class TrainingHistory:
    def __init__(self, model_name):
        self.model_name = model_name
        self.epoch_data = []
        self.boosting_results = {}
        
    def add_epoch(self, epoch, train_loss, val_loss, train_metrics, val_metrics, lr, epoch_time):
        epoch_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_metrics['accuracy'],
            'val_accuracy': val_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'val_precision': val_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_recall': val_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'val_f1': val_metrics['f1'],
            'train_auroc': train_metrics.get('auroc', 0),
            'val_auroc': val_metrics.get('auroc', 0),
            'lr': lr,
            'time': epoch_time
        }
        self.epoch_data.append(epoch_info)
        
    def add_boosting_result(self, method, metrics):
        self.boosting_results[method] = metrics
        
    def print_epoch_results(self, epoch):
        if epoch < len(self.epoch_data):
            data = self.epoch_data[epoch]
            print(f"\n📊 EPOCH {data['epoch']:2d} RESULTS:")
            print(f"   Loss:      Train={data['train_loss']:.4f} | Val={data['val_loss']:.4f}")
            print(f"   Accuracy:  Train={data['train_accuracy']:.4f} | Val={data['val_accuracy']:.4f}")
            print(f"   Precision: Train={data['train_precision']:.4f} | Val={data['val_precision']:.4f}")
            print(f"   Recall:    Train={data['train_recall']:.4f} | Val={data['val_recall']:.4f}")
            print(f"   F1-Score:  Train={data['train_f1']:.4f} | Val={data['val_f1']:.4f}")
            print(f"   AUC-ROC:   Train={data['train_auroc']:.4f} | Val={data['val_auroc']:.4f}")
            print(f"   LR: {data['lr']:.2e} | Time: {data['time']:.2f}s")
        
    def plot_training_history(self):
        if not self.epoch_data:
            print("❌ No training data to plot")
            return
            
        epochs = [data['epoch'] for data in self.epoch_data]
        train_losses = [data['train_loss'] for data in self.epoch_data]
        val_losses = [data['val_loss'] for data in self.epoch_data]
        train_accs = [data['train_accuracy'] for data in self.epoch_data]
        val_accs = [data['val_accuracy'] for data in self.epoch_data]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_title(f'{self.model_name} - Training & Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
        ax2.set_title(f'{self.model_name} - Training & Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score plot
        train_f1s = [data['train_f1'] for data in self.epoch_data]
        val_f1s = [data['val_f1'] for data in self.epoch_data]
        ax3.plot(epochs, train_f1s, 'b-', label='Train F1', linewidth=2)
        ax3.plot(epochs, val_f1s, 'r-', label='Val F1', linewidth=2)
        ax3.set_title(f'{self.model_name} - Training & Validation F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate plot
        lrs = [data['lr'] for data in self.epoch_data]
        ax4.plot(epochs, lrs, 'g-', label='Learning Rate', linewidth=2)
        ax4.set_title(f'{self.model_name} - Learning Rate Schedule')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

# Metrics Calculation
def calculate_detailed_metrics(true_labels, predictions, probs=None):
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

# Fixed Training Function
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_probs = []
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} Training")):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if device.type == 'cuda' and use_mixed_precision:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        # FIX: Use detach() before converting to numpy
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
        
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_detailed_metrics(all_labels, all_predictions, all_probs)
    return avg_loss, metrics

# Fixed Evaluation Function
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
            # FIX: Use detach() before converting to numpy
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    all_probs = np.array(all_probs)
    metrics = calculate_detailed_metrics(all_labels, all_predictions, all_probs)
    return avg_loss, metrics

# Fixed Feature Extraction Function
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feat = model(images, features_only=True)
            # FIX: Use detach() before converting to numpy
            features.append(feat.detach().cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

# BOHB Optimization
def run_bohb_optimization(X_train, y_train, X_val, y_val, X_test, y_test, history):
    print("\n" + "="*60)
    print("🔧 BOHB OPTIMIZATION FOR XGBoost")
    print("="*60)
    
    try:
        class LocalBOHBWorker(Worker):
            def __init__(self, X_train, y_train, X_val, y_val, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.X_train = X_train
                self.y_train = y_train
                self.X_val = X_val
                self.y_val = y_val
                
            def compute(self, config, budget, **kwargs):
                model = xgb.XGBClassifier(
                    n_estimators=int(budget),
                    learning_rate=config['learning_rate'],
                    max_depth=config['max_depth'],
                    subsample=config['subsample'],
                    colsample_bytree=config['colsample_bytree'],
                    random_state=42
                )
                model.fit(self.X_train, self.y_train)
                val_pred = model.predict(self.X_val)
                f1 = f1_score(self.y_val, val_pred, average='weighted')
                return {'loss': 1 - f1, 'info': {'f1_score': f1}}
        
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', 0.01, 0.3))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_depth', 3, 10))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('subsample', 0.5, 1.0))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('colsample_bytree', 0.5, 1.0))
        
        worker = LocalBOHBWorker(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, run_id='bohb_xgb')
        
        bohb = BOHB(
            configspace=config_space,
            run_id='bohb_xgb',
            min_budget=50,
            max_budget=150,
            worker=worker
        )
        
        print("Starting BOHB optimization...")
        res = bohb.run(n_iterations=5)
        
        best_config = res.get_incumbent_id()
        best_loss = res.get_incumbent_loss()
        
        print(f"🎯 BOHB Best Configuration:")
        print(f"   Learning Rate: {best_config['learning_rate']:.4f}")
        print(f"   Max Depth: {best_config['max_depth']}")
        print(f"   Subsample: {best_config['subsample']:.3f}")
        print(f"   Colsample: {best_config['colsample_bytree']:.3f}")
        print(f"   Best Loss: {best_loss:.4f}")
        
        # Train final model
        best_xgb = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=best_config['learning_rate'],
            max_depth=best_config['max_depth'],
            subsample=best_config['subsample'],
            colsample_bytree=best_config['colsample_bytree'],
            random_state=42
        )
        best_xgb.fit(X_train, y_train)
        
        # Evaluate
        xgb_preds = {
            'train': best_xgb.predict(X_train),
            'val': best_xgb.predict(X_val),
            'test': best_xgb.predict(X_test)
        }
        xgb_probs = {
            'train': best_xgb.predict_proba(X_train),
            'val': best_xgb.predict_proba(X_val),
            'test': best_xgb.predict_proba(X_test)
        }
        
        bohb_metrics = {
            'train': calculate_detailed_metrics(y_train, xgb_preds['train'], xgb_probs['train']),
            'val': calculate_detailed_metrics(y_val, xgb_preds['val'], xgb_probs['val']),
            'test': calculate_detailed_metrics(y_test, xgb_preds['test'], xgb_probs['test'])
        }
        
        history.add_boosting_result('XGBoost_BOHB', bohb_metrics)
        
        print("\n📊 BOHB-Optimized XGBoost Results:")
        print(f"Train - Acc: {bohb_metrics['train']['accuracy']:.4f}, F1: {bohb_metrics['train']['f1']:.4f}")
        print(f"Val   - Acc: {bohb_metrics['val']['accuracy']:.4f}, F1: {bohb_metrics['val']['f1']:.4f}")
        print(f"Test  - Acc: {bohb_metrics['test']['accuracy']:.4f}, F1: {bohb_metrics['test']['f1']:.4f}")
        
    except Exception as e:
        print(f"❌ BOHB optimization failed: {e}")
        print("Using default XGBoost as fallback...")
        default_xgb = xgb.XGBClassifier(random_state=42)
        default_xgb.fit(X_train, y_train)
        
        xgb_preds = {
            'train': default_xgb.predict(X_train),
            'val': default_xgb.predict(X_val),
            'test': default_xgb.predict(X_test)
        }
        xgb_probs = {
            'train': default_xgb.predict_proba(X_train),
            'val': default_xgb.predict_proba(X_val),
            'test': default_xgb.predict_proba(X_test)
        }
        
        default_metrics = {
            'train': calculate_detailed_metrics(y_train, xgb_preds['train'], xgb_probs['train']),
            'val': calculate_detailed_metrics(y_val, xgb_preds['val'], xgb_probs['val']),
            'test': calculate_detailed_metrics(y_test, xgb_preds['test'], xgb_probs['test'])
        }
        
        history.add_boosting_result('XGBoost_Default', default_metrics)

# Optuna Optimization
def run_optuna_optimization(X_train, y_train, X_val, y_val, X_test, y_test, history):
    print("\n" + "="*60)
    print("🔧 OPTUNA OPTIMIZATION FOR XGBoost")
    print("="*60)
    
    def optuna_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        return f1_score(y_val, val_pred, average='weighted')
    
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(optuna_objective, n_trials=20)
        
        print(f"🎯 Optuna Best Params: {study.best_params}")
        print(f"🎯 Optuna Best Value: {study.best_value:.4f}")
        
        best_xgb = xgb.XGBClassifier(**study.best_params, random_state=42)
        best_xgb.fit(X_train, y_train)
        
        xgb_preds = {
            'train': best_xgb.predict(X_train),
            'val': best_xgb.predict(X_val),
            'test': best_xgb.predict(X_test)
        }
        xgb_probs = {
            'train': best_xgb.predict_proba(X_train),
            'val': best_xgb.predict_proba(X_val),
            'test': best_xgb.predict_proba(X_test)
        }
        
        optuna_metrics = {
            'train': calculate_detailed_metrics(y_train, xgb_preds['train'], xgb_probs['train']),
            'val': calculate_detailed_metrics(y_val, xgb_preds['val'], xgb_probs['val']),
            'test': calculate_detailed_metrics(y_test, xgb_preds['test'], xgb_probs['test'])
        }
        
        history.add_boosting_result('XGBoost_Optuna', optuna_metrics)
        
        print("\n📊 Optuna-Optimized XGBoost Results:")
        print(f"Train - Acc: {optuna_metrics['train']['accuracy']:.4f}, F1: {optuna_metrics['train']['f1']:.4f}")
        print(f"Val   - Acc: {optuna_metrics['val']['accuracy']:.4f}, F1: {optuna_metrics['val']['f1']:.4f}")
        print(f"Test  - Acc: {optuna_metrics['test']['accuracy']:.4f}, F1: {optuna_metrics['test']['f1']:.4f}")
        
    except Exception as e:
        print(f"❌ Optuna optimization failed: {e}")

# Main Execution
def main():
    print("🚀 HYBRID MODEL 1: EfficientNetV2M + MobileViT")
    print("="*70)
    
    # 1. Data Loading
    print("\n📁 STEP 1: LOADING DATASET...")
    data_dir = "/Volumes/T7/KFU/Nov14/KFU_Dataset"
    dataset = FoodDataset(data_dir, transform=train_transform, is_training=True)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Apply validation transform to val/test sets
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Data loaders
    batch_size = 16
    num_workers = 0
    pin_memory = False
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    
    print(f"✅ Dataset split:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # 2. Model Initialization
    print("\n🤖 STEP 2: INITIALIZING HYBRID MODEL...")
    model = HybridModel1(num_classes=2).to(device)
    history = TrainingHistory("EfficientNetV2M_MobileViT")
    
    # Training configuration
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    total_epochs = 10
    efficientnet_epochs = 5
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    
    # 3. Training Loop
    print("\n🎯 STEP 3: STARTING TRAINING...")
    print("="*70)
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        
        # Phase training
        if epoch < efficientnet_epochs:
            # Freeze MobileViT, train EfficientNetV2M
            for param in model.mobilevit.parameters():
                param.requires_grad = False
            for param in model.efficientnet.parameters():
                param.requires_grad = True
            phase = "EfficientNetV2M"
        else:
            # Freeze EfficientNetV2M, train MobileViT
            for param in model.efficientnet.parameters():
                param.requires_grad = False
            for param in model.mobilevit.parameters():
                param.requires_grad = True
            phase = "MobileViT"
        
        print(f"\n📈 Epoch {epoch+1}/{total_epochs} - Training {phase}")
        
        # Training
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        
        # Validation
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Store detailed epoch results
        history.add_epoch(epoch+1, train_loss, val_loss, train_metrics, val_metrics, current_lr, epoch_time)
        history.print_epoch_results(epoch)
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_hybrid_model1.pth')
            print(f"✅ New best model saved! Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "="*70)
    print("🎯 DEEP LEARNING TRAINING COMPLETED!")
    print("="*70)
    
    # 4. Feature Extraction
    print("\n🔍 STEP 4: EXTRACTING FEATURES...")
    model.load_state_dict(torch.load('best_hybrid_model1.pth'))
    X_train, y_train = extract_features(model, train_loader, device)
    X_val, y_val = extract_features(model, val_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)
    
    print(f"✅ Feature shapes:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Save features
    np.save('X_train_features.npy', X_train)
    np.save('y_train_labels.npy', y_train)
    np.save('X_val_features.npy', X_val)
    np.save('y_val_labels.npy', y_val)
    np.save('X_test_features.npy', X_test)
    np.save('y_test_labels.npy', y_test)
    print("💾 Features saved to disk")
    
    # 5. XGBoost Training
    print("\n🌲 STEP 5: TRAINING XGBOOST...")
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Evaluate XGBoost
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_train_proba = xgb_model.predict_proba(X_train)
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_val_proba = xgb_model.predict_proba(X_val)
    xgb_test_pred = xgb_model.predict(X_test)
    xgb_test_proba = xgb_model.predict_proba(X_test)
    
    xgb_metrics = {
        'train': calculate_detailed_metrics(y_train, xgb_train_pred, xgb_train_proba),
        'val': calculate_detailed_metrics(y_val, xgb_val_pred, xgb_val_proba),
        'test': calculate_detailed_metrics(y_test, xgb_test_pred, xgb_test_proba)
    }
    
    history.add_boosting_result('XGBoost_Default', xgb_metrics)
    
    print("\n📊 Default XGBoost Results:")
    print(f"Train - Acc: {xgb_metrics['train']['accuracy']:.4f}, F1: {xgb_metrics['train']['f1']:.4f}")
    print(f"Val   - Acc: {xgb_metrics['val']['accuracy']:.4f}, F1: {xgb_metrics['val']['f1']:.4f}")
    print(f"Test  - Acc: {xgb_metrics['test']['accuracy']:.4f}, F1: {xgb_metrics['test']['f1']:.4f}")
    
    # 6. Optimization
    print("\n🔧 STEP 6: RUNNING OPTIMIZATIONS...")
    run_bohb_optimization(X_train, y_train, X_val, y_val, X_test, y_test, history)
    run_optuna_optimization(X_train, y_train, X_val, y_val, X_test, y_test, history)
    
    # 7. Visualization
    print("\n📊 STEP 7: GENERATING VISUALIZATIONS...")
    history.plot_training_history()
    
    # Final Summary
    print("\n" + "="*70)
    print("🎉 HYBRID MODEL 1 - FINAL SUMMARY")
    print("="*70)
    print("Architecture: EfficientNetV2M + MobileViT")
    print("\nBoosting Classifiers Performance:")
    for method, results in history.boosting_results.items():
        print(f"\n{method}:")
        print(f"  Train - Acc: {results['train']['accuracy']:.4f}, F1: {results['train']['f1']:.4f}")
        print(f"  Val   - Acc: {results['val']['accuracy']:.4f}, F1: {results['val']['f1']:.4f}")
        print(f"  Test  - Acc: {results['test']['accuracy']:.4f}, F1: {results['test']['f1']:.4f}")
    
    print(f"\n✅ Hybrid Model 1 training completed!")
    print(f"📊 Visualizations saved as: EfficientNetV2M_MobileViT_training_history.png")

if __name__ == "__main__":
    main()
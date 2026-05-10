# SHAP Explainer for Hybrid Model 1
# Using saved model and features for comprehensive explainable AI

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import pandas as pd
import warnings
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import cv2
import json
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

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

# Import the model architecture from your original file
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define minimal model architecture for loading
class EfficientNetV2FeatureExtractor(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
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
        import timm
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
            return fused_feat
        return output

class HybridSHAPExplainer:
    def __init__(self, base_dir, results_dir="hybrid_SHAP_1"):
        """
        Initialize SHAP Explainer for Hybrid Model
        
        Args:
            base_dir: Base directory containing project files
            results_dir: Directory containing saved results
        """
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, results_dir)
        self.device = device
        self.class_names = ["Bad Quality", "Good Quality"]
        
        # Create output directories
        self.output_dir = os.path.join(self.results_dir, "shap_visualizations")
        self.dirs_to_create = ["summary_plots", "force_plots", "heatmaps", 
                               "overlays", "feature_importance", "reports"]
        
        for dir_name in self.dirs_to_create:
            os.makedirs(os.path.join(self.output_dir, dir_name), exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
        
        # Load model and data
        self.model = self.load_model()
        self.xgboost_model = self.load_xgboost_model()
        self.features, self.labels = self.load_features()
        
    def load_model(self):
        """Load the trained Hybrid Model 1"""
        print("🤖 Loading Hybrid Model 1...")
        
        model_path = os.path.join(self.results_dir, "best_hybrid_model1.pth")
        
        if not os.path.exists(model_path):
            # Search in parent directory
            model_path = os.path.join(self.base_dir, "best_hybrid_model1.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Initialize model
        model = HybridModel1(num_classes=2).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        print(f"✅ Model loaded from: {model_path}")
        return model
    
    def load_xgboost_model(self):
        """Load or train XGBoost model"""
        print("🌲 Loading XGBoost model...")
        
        xgb_path = os.path.join(self.results_dir, "xgboost_model.pkl")
        
        if os.path.exists(xgb_path):
            try:
                with open(xgb_path, 'rb') as f:
                    xgb_model = pickle.load(f)
                print(f"✅ XGBoost model loaded from: {xgb_path}")
                return xgb_model
            except:
                print("⚠️ Could not load XGBoost model, will train new one")
        
        # If not found, we'll train it later
        return None
    
    def load_features(self):
        """Load saved features"""
        print("📊 Loading saved features...")
        
        feature_files = {
            'X_train': 'X_train_features.npy',
            'X_val': 'X_val_features.npy',
            'X_test': 'X_test_features.npy',
            'y_train': 'y_train_labels.npy',
            'y_val': 'y_val_labels.npy',
            'y_test': 'y_test_labels.npy'
        }
        
        loaded_features = {}
        
        for key, filename in feature_files.items():
            file_path = os.path.join(self.results_dir, filename)
            if os.path.exists(file_path):
                loaded_features[key] = np.load(file_path)
                print(f"✅ Loaded {key}: {loaded_features[key].shape}")
            else:
                print(f"⚠️ {filename} not found at {file_path}")
        
        # Combine all features for SHAP analysis
        if 'X_train' in loaded_features and 'X_test' in loaded_features:
            X_combined = np.vstack([loaded_features['X_train'], 
                                   loaded_features['X_val'], 
                                   loaded_features['X_test']])
            y_combined = np.concatenate([loaded_features['y_train'], 
                                        loaded_features['y_val'], 
                                        loaded_features['y_test']])
            print(f"✅ Combined features shape: {X_combined.shape}")
            print(f"✅ Combined labels shape: {y_combined.shape}")
            return X_combined, y_combined
        else:
            raise ValueError("Required feature files not found!")
    
    def create_xgboost_shap_explanations(self, num_samples=1000):
        """Create SHAP explanations using XGBoost model"""
        print("\n" + "="*60)
        print("🌲 XGBoost SHAP Analysis")
        print("="*60)
        
        # Train XGBoost if not loaded
        if self.xgboost_model is None:
            print("Training XGBoost model...")
            self.xgboost_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.labels, test_size=0.2, random_state=42
            )
            
            self.xgboost_model.fit(X_train, y_train)
            
            # Save the model
            xgb_path = os.path.join(self.results_dir, "xgboost_model.pkl")
            with open(xgb_path, 'wb') as f:
                pickle.dump(self.xgboost_model, f)
            print(f"✅ XGBoost model saved to: {xgb_path}")
        
        # Sample data for faster SHAP computation
        if len(self.features) > num_samples:
            indices = np.random.choice(len(self.features), num_samples, replace=False)
            X_sample = self.features[indices]
            y_sample = self.labels[indices]
        else:
            X_sample = self.features
            y_sample = self.labels
        
        print(f"Using {len(X_sample)} samples for SHAP analysis")
        
        # Create SHAP explainer
        print("Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(self.xgboost_model)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        
        # Convert to numpy array if needed
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        print(f"SHAP values shape: {shap_values.shape}")
        
        # 1. Create SHAP Summary Plot
        print("\n📊 Creating SHAP summary plot...")
        plt.figure(figsize=(15, 10))
        
        # For binary classification, we have SHAP values for both classes
        if len(shap_values.shape) == 3:  # [samples, features, classes]
            # Use SHAP values for positive class (class 1)
            shap.summary_plot(
                shap_values[:, :, 1], 
                X_sample,
                feature_names=[f"Feature_{i}" for i in range(X_sample.shape[1])],
                show=False,
                max_display=20,
                plot_size=(12, 8)
            )
        else:
            shap.summary_plot(
                shap_values, 
                X_sample,
                feature_names=[f"Feature_{i}" for i in range(X_sample.shape[1])],
                show=False,
                max_display=20,
                plot_size=(12, 8)
            )
        
        plt.title("SHAP Feature Importance Summary", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        summary_path = os.path.join(self.output_dir, "summary_plots", "shap_summary.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Summary plot saved: {summary_path}")
        
        # 2. Calculate and visualize Mean Absolute SHAP
        print("\n📈 Calculating Mean Absolute SHAP values...")
        if len(shap_values.shape) == 3:
            mean_abs_shap = np.mean(np.abs(shap_values[:, :, 1]), axis=0)  # For positive class
        else:
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create feature names
        feature_names = [f"Feature_{i}" for i in range(len(mean_abs_shap))]
        
        # Sort features by importance
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        top_n = min(30, len(mean_abs_shap))
        
        # Create bar plot
        plt.figure(figsize=(14, 8))
        plt.barh(range(top_n), mean_abs_shap[sorted_idx[:top_n]][::-1])
        plt.yticks(range(top_n), [feature_names[i] for i in sorted_idx[:top_n]][::-1])
        plt.xlabel('Mean Absolute SHAP Value', fontsize=14)
        plt.title(f'Top {top_n} Most Important Features\n(Mean |SHAP|)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        mean_shap_path = os.path.join(self.output_dir, "feature_importance", "mean_absolute_shap.png")
        plt.savefig(mean_shap_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Mean absolute SHAP plot saved: {mean_shap_path}")
        
        # Save SHAP values to CSV
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'mean_absolute_shap': mean_abs_shap,
            'feature_index': range(len(mean_abs_shap))
        })
        shap_df = shap_df.sort_values('mean_absolute_shap', ascending=False)
        
        csv_path = os.path.join(self.output_dir, "feature_importance", "shap_values.csv")
        shap_df.to_csv(csv_path, index=False)
        print(f"✅ SHAP values saved to CSV: {csv_path}")
        
        # 3. Create Force Plots for sample predictions
        print("\n🎯 Creating SHAP force plots...")
        
        # Get predictions
        y_pred = self.xgboost_model.predict(X_sample)
        y_pred_proba = self.xgboost_model.predict_proba(X_sample)
        
        # Create force plots for some interesting samples
        interesting_samples = []
        
        # Find samples where model is confident but wrong
        for i in range(min(100, len(X_sample))):
            proba = y_pred_proba[i]
            confidence = max(proba)
            pred = y_pred[i]
            true = y_sample[i]
            
            if pred != true and confidence > 0.7:  # Wrong but confident
                interesting_samples.append((i, "wrong_confident", confidence))
            elif pred == true and confidence > 0.9:  # Correct and very confident
                interesting_samples.append((i, "correct_confident", confidence))
            elif 0.4 < confidence < 0.6:  # Uncertain
                interesting_samples.append((i, "uncertain", confidence))
        
        # Sort by interestingness
        interesting_samples.sort(key=lambda x: x[2], reverse=True)
        
        # Create force plots for top interesting samples
        for idx, (sample_idx, sample_type, confidence) in enumerate(interesting_samples[:5]):
            print(f"  Creating force plot for sample {sample_idx} ({sample_type}, confidence={confidence:.2%})...")
            
            plt.figure(figsize=(12, 4))
            
            if len(shap_values.shape) == 3:
                # For binary classification
                shap.force_plot(
                    explainer.expected_value[1],  # Expected value for positive class
                    shap_values[sample_idx, :, 1],  # SHAP values for positive class
                    X_sample[sample_idx],
                    feature_names=[f"F{i}" for i in range(X_sample.shape[1])],
                    matplotlib=True,
                    show=False
                )
            else:
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[sample_idx, :],
                    X_sample[sample_idx],
                    feature_names=[f"F{i}" for i in range(X_sample.shape[1])],
                    matplotlib=True,
                    show=False
                )
            
            plt.title(f"Force Plot - Sample {sample_idx}\n"
                     f"True: {self.class_names[y_sample[sample_idx]]}, "
                     f"Pred: {self.class_names[y_pred[sample_idx]]}, "
                     f"Confidence: {confidence:.2%}",
                     fontsize=12, fontweight='bold', pad=20)
            plt.tight_layout()
            
            force_path = os.path.join(self.output_dir, "force_plots", 
                                     f"force_plot_sample_{sample_idx}_{sample_type}.png")
            plt.savefig(force_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            print(f"    ✅ Force plot saved: {force_path}")
        
        # 4. Create Dependence Plots for top features
        print("\n🔗 Creating SHAP dependence plots...")
        
        top_features = sorted_idx[:5]  # Top 5 features
        
        for i, feature_idx in enumerate(top_features):
            plt.figure(figsize=(10, 6))
            
            if len(shap_values.shape) == 3:
                shap.dependence_plot(
                    feature_idx,
                    shap_values[:, :, 1],  # For positive class
                    X_sample,
                    feature_names=[f"F{i}" for i in range(X_sample.shape[1])],
                    show=False
                )
            else:
                shap.dependence_plot(
                    feature_idx,
                    shap_values,
                    X_sample,
                    feature_names=[f"F{i}" for i in range(X_sample.shape[1])],
                    show=False
                )
            
            plt.title(f"SHAP Dependence Plot - Feature {feature_idx}\n{feature_names[feature_idx]}",
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            dep_path = os.path.join(self.output_dir, "summary_plots", 
                                   f"dependence_plot_feature_{feature_idx}.png")
            plt.savefig(dep_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"  ✅ Dependence plot saved: {dep_path}")
        
        return shap_values, mean_abs_shap
    
    def create_cnn_heatmaps(self, num_samples=5):
        """Create Grad-CAM heatmaps for CNN model"""
        print("\n" + "="*60)
        print("🔥 CNN Grad-CAM Heatmaps")
        print("="*60)
        
        # We need to load actual images for Grad-CAM
        # First, let's find some sample images from the dataset
        data_dir = os.path.join(self.base_dir, "Master Dataset")
        
        if not os.path.exists(data_dir):
            print(f"⚠️ Dataset directory not found: {data_dir}")
            print("Skipping Grad-CAM visualizations...")
            return
        
        print(f"📁 Loading sample images from: {data_dir}")
        
        # We'll create a simple function to find sample images
        def find_sample_images(num_samples=5):
            import glob
            images = []
            
            # Search for images in the dataset
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.heic']
            
            for ext in image_extensions:
                pattern = os.path.join(data_dir, '**', ext)
                found_images = glob.glob(pattern, recursive=True)
                images.extend(found_images)
                
                if len(images) >= num_samples * 2:  # Get extra for safety
                    break
            
            return images[:num_samples]
        
        sample_images = find_sample_images(num_samples)
        
        if not sample_images:
            print("⚠️ No sample images found!")
            return
        
        print(f"Found {len(sample_images)} sample images")
        
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Process each sample image
        for idx, img_path in enumerate(sample_images[:num_samples]):
            print(f"\n  Processing image {idx+1}/{min(num_samples, len(sample_images))}: {os.path.basename(img_path)}")
            
            try:
                # Load and preprocess image
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                
                # Get model prediction
                with torch.no_grad():
                    output = self.model(image_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred_class = torch.argmax(output, dim=1).item()
                    confidence = probs[0, pred_class].item()
                
                print(f"    Prediction: {self.class_names[pred_class]} ({confidence:.2%})")
                
                # Generate Grad-CAM heatmap
                heatmap = self.generate_gradcam(image_tensor, pred_class)
                
                # Create visualization
                self.create_heatmap_visualization(
                    image, heatmap, pred_class, confidence, idx, img_path
                )
                
            except Exception as e:
                print(f"    ⚠️ Error processing image: {e}")
    
    def generate_gradcam(self, image_tensor, target_class):
        """Generate Grad-CAM heatmap"""
        # Register hooks for gradients and activations
        gradients = []
        activations = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
        
        # Hook the last convolutional layer of EfficientNet
        target_layer = self.model.efficientnet.backbone.features[-1]
        backward_handle = target_layer.register_backward_hook(backward_hook)
        forward_handle = target_layer.register_forward_hook(forward_hook)
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Remove hooks
        backward_handle.remove()
        forward_handle.remove()
        
        if not gradients or not activations:
            return None
        
        # Compute Grad-CAM
        grads = gradients[0]
        acts = activations[0]
        
        # Global average pooling of gradients
        pooled_grads = torch.mean(grads, dim=[0, 2, 3])
        
        # Weight activations by gradients
        for i in range(acts.size(1)):
            acts[:, i, :, :] *= pooled_grads[i]
        
        # Generate heatmap
        heatmap = torch.mean(acts, dim=1).squeeze()
        heatmap = torch.relu(heatmap)  # ReLU to keep only positive influences
        heatmap = heatmap / (heatmap.max() + 1e-8)  # Normalize
        
        return heatmap.cpu().numpy()
    
    def create_heatmap_visualization(self, original_image, heatmap, pred_class, confidence, idx, img_path):
        """Create heatmap overlay visualization"""
        # Convert PIL image to numpy
        img_np = np.array(original_image.resize((224, 224)))
        
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        
        # Normalize heatmap to 0-255
        heatmap_normalized = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap\n(Red = Important regions)", 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\nPrediction: {self.class_names[pred_class]}\nConfidence: {confidence:.2%}",
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f"Sample {idx} - {os.path.basename(img_path)}", 
                    fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        # Save figure
        heatmap_path = os.path.join(self.output_dir, "heatmaps", f"heatmap_sample_{idx}.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"    ✅ Heatmap visualization saved: {heatmap_path}")
        
        # Save overlay separately
        overlay_path = os.path.join(self.output_dir, "overlays", f"overlay_sample_{idx}.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"    ✅ Overlay image saved: {overlay_path}")
    
    def generate_comprehensive_report(self, shap_values, mean_abs_shap):
        """Generate comprehensive explainability report"""
        print("\n" + "="*60)
        print("📋 Generating Comprehensive Report")
        print("="*60)
        
        report_data = {
            "model_info": {
                "model_name": "HybridModel1 (EfficientNetV2M + MobileViT)",
                "device": str(self.device),
                "classes": self.class_names,
                "total_samples": len(self.features)
            },
            "shap_analysis": {
                "total_features": len(mean_abs_shap),
                "mean_abs_shap_mean": float(np.mean(mean_abs_shap)),
                "mean_abs_shap_std": float(np.std(mean_abs_shap)),
                "mean_abs_shap_max": float(np.max(mean_abs_shap)),
                "mean_abs_shap_min": float(np.min(mean_abs_shap)),
                "top_10_features": []
            },
            "visualizations": {
                "summary_plot": "shap_summary.png",
                "mean_abs_shap": "mean_absolute_shap.png",
                "force_plots": [],
                "heatmaps": [],
                "overlays": []
            }
        }
        
        # Get top 10 features
        top_indices = np.argsort(mean_abs_shap)[::-1][:10]
        for i, idx in enumerate(top_indices):
            report_data["shap_analysis"]["top_10_features"].append({
                "rank": i + 1,
                "feature_index": int(idx),
                "feature_name": f"Feature_{idx}",
                "mean_abs_shap": float(mean_abs_shap[idx])
            })
        
        # List visualizations
        force_plot_dir = os.path.join(self.output_dir, "force_plots")
        if os.path.exists(force_plot_dir):
            report_data["visualizations"]["force_plots"] = [
                f for f in os.listdir(force_plot_dir) if f.endswith('.png')
            ]
        
        heatmap_dir = os.path.join(self.output_dir, "heatmaps")
        if os.path.exists(heatmap_dir):
            report_data["visualizations"]["heatmaps"] = [
                f for f in os.listdir(heatmap_dir) if f.endswith('.png')
            ]
        
        overlay_dir = os.path.join(self.output_dir, "overlays")
        if os.path.exists(overlay_dir):
            report_data["visualizations"]["overlays"] = [
                f for f in os.listdir(overlay_dir) if f.endswith('.png')
            ]
        
        # Save JSON report
        report_path = os.path.join(self.output_dir, "reports", "shap_analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"✅ JSON report saved: {report_path}")
        
        # Create markdown report
        md_report = self._create_markdown_report(report_data)
        md_path = os.path.join(self.output_dir, "reports", "shap_analysis_report.md")
        with open(md_path, 'w') as f:
            f.write(md_report)
        
        print(f"✅ Markdown report saved: {md_path}")
        
        # Print summary
        print("\n📊 SHAP Analysis Summary:")
        print(f"   - Top feature: Feature_{top_indices[0]} (Mean |SHAP| = {mean_abs_shap[top_indices[0]]:.6f})")
        print(f"   - Average feature importance: {np.mean(mean_abs_shap):.6f}")
        print(f"   - Total features analyzed: {len(mean_abs_shap)}")
        print(f"   - Visualizations created: {len(report_data['visualizations']['force_plots'])} force plots")
        print(f"   - Visualizations created: {len(report_data['visualizations']['heatmaps'])} heatmaps")
    
    def _create_markdown_report(self, report_data):
        """Create markdown report from report data"""
        md = f"""# SHAP Analysis Report - Hybrid Model 1

## Model Information
- **Model Name**: {report_data['model_info']['model_name']}
- **Device**: {report_data['model_info']['device']}
- **Classes**: {', '.join(report_data['model_info']['classes'])}
- **Total Samples**: {report_data['model_info']['total_samples']:,}

## SHAP Analysis Results
- **Total Features Analyzed**: {report_data['shap_analysis']['total_features']}
- **Mean Absolute SHAP**: {report_data['shap_analysis']['mean_abs_shap_mean']:.6f}
- **Standard Deviation**: {report_data['shap_analysis']['mean_abs_shap_std']:.6f}
- **Maximum Value**: {report_data['shap_analysis']['mean_abs_shap_max']:.6f}
- **Minimum Value**: {report_data['shap_analysis']['mean_abs_shap_min']:.6f}

## Top 10 Most Important Features
| Rank | Feature Index | Feature Name | Mean Absolute SHAP |
|------|---------------|--------------|---------------------|
"""
        
        for feature in report_data['shap_analysis']['top_10_features']:
            md += f"| {feature['rank']} | {feature['feature_index']} | {feature['feature_name']} | {feature['mean_abs_shap']:.6f} |\n"
        
        md += """

## Generated Visualizations

### Summary Plots
- `shap_summary.png` - SHAP summary plot showing feature importance
- `mean_absolute_shap.png` - Bar plot of mean absolute SHAP values

### Force Plots
"""
        
        for plot in report_data['visualizations']['force_plots']:
            md += f"- `{plot}`\n"
        
        md += """

### Heatmaps
"""
        
        for heatmap in report_data['visualizations']['heatmaps']:
            md += f"- `{heatmap}`\n"
        
        md += """

### Overlays
"""
        
        for overlay in report_data['visualizations']['overlays']:
            md += f"- `{overlay}`\n"
        
        md += """

## Interpretation Guide

### SHAP Values
- **Positive SHAP values**: Push prediction toward GOOD quality
- **Negative SHAP values**: Push prediction toward BAD quality
- **Higher absolute values**: More important features

### Heatmap Colors
- **Red regions**: Areas that strongly suggest GOOD quality
- **Blue regions**: Areas that suggest BAD quality
- **Yellow/Green regions**: Moderate influence on prediction

### Model Trust Indicators
1. **Consistent heatmap patterns** across similar images = High trust
2. **Heatmaps focusing on food regions** = Good alignment with task
3. **Scattered or irrelevant heatmaps** = Potential issues with model

---

*Report generated automatically by HybridSHAPExplainer*
"""
        
        return md
    
    def run_complete_analysis(self, num_shap_samples=1000, num_heatmap_samples=3):
        """Run complete SHAP analysis pipeline"""
        print("="*70)
        print("🔍 COMPLETE SHAP ANALYSIS PIPELINE")
        print("="*70)
        
        try:
            # 1. XGBoost SHAP Analysis
            print("\n📊 Phase 1: XGBoost SHAP Analysis")
            shap_values, mean_abs_shap = self.create_xgboost_shap_explanations(
                num_samples=num_shap_samples
            )
            
            # 2. CNN Heatmap Analysis
            print("\n📊 Phase 2: CNN Heatmap Analysis")
            self.create_cnn_heatmaps(num_samples=num_heatmap_samples)
            
            # 3. Generate Comprehensive Report
            print("\n📊 Phase 3: Report Generation")
            self.generate_comprehensive_report(shap_values, mean_abs_shap)
            
            print("\n" + "="*70)
            print("✅ SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*70)
            
            print(f"\n📁 All results saved to: {self.output_dir}")
            print("\n📊 Key Results Generated:")
            print("   1. SHAP Summary Plot")
            print("   2. Mean Absolute SHAP Bar Plot")
            print("   3. SHAP Force Plots (individual predictions)")
            print("   4. SHAP Dependence Plots")
            print("   5. Grad-CAM Heatmaps")
            print("   6. Heatmap Overlays on Images")
            print("   7. Comprehensive JSON Report")
            print("   8. Markdown Summary Report")
            
            # Show sample of top features
            top_indices = np.argsort(mean_abs_shap)[::-1][:5]
            print(f"\n🏆 Top 5 Most Important Features:")
            for i, idx in enumerate(top_indices):
                print(f"   {i+1}. Feature_{idx}: Mean |SHAP| = {mean_abs_shap[idx]:.6f}")
            
        except Exception as e:
            print(f"\n❌ Error during SHAP analysis: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function"""
    print("🔍 HYBRID MODEL 1 - SHAP EXPLAINER")
    print("="*70)
    
    # Configuration - UPDATE THESE PATHS
    BASE_DIR = "/Users/tanaypatel/Documents/projects/Machine_Learning/KFU/food_classification_project"
    RESULTS_DIR = "/Users/tanaypatel/Documents/projects/Machine_Learning/KFU/food_classification_project/results/hybrid_SHAP_1"  # Directory containing saved results
    
    # Initialize SHAP explainer
    explainer = HybridSHAPExplainer(
        base_dir=BASE_DIR,
        results_dir=RESULTS_DIR
    )
    
    # Run complete analysis
    # Adjust these parameters based on your system's memory:
    # - num_shap_samples: Number of samples for SHAP analysis (higher = more accurate but slower)
    # - num_heatmap_samples: Number of images for Grad-CAM heatmaps
    explainer.run_complete_analysis(
        num_shap_samples=500,    # Start with 500, increase if you have more memory
        num_heatmap_samples=3    # 3 heatmaps is usually sufficient
    )

if __name__ == "__main__":
    # Check if SHAP is installed
    try:
        import shap
        print("✅ SHAP is available")
    except ImportError:
        print("❌ SHAP is not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
        import shap
        print("✅ SHAP installed successfully")
    
    main()
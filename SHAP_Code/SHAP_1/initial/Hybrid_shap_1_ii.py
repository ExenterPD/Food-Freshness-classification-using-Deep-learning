# SHAP Analysis for Hybrid Model 1: EfficientNetV2M + MobileViT
# Complete Working Version with Force Plots and Increased Samples

import torch
import torch.nn as nn
import numpy as np
import shap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import seaborn as sns
import warnings
import os
import sys
from tqdm import tqdm
import json
from sklearn.metrics import confusion_matrix

# Import your model architecture
sys.path.append('.')  # Add current directory to path
from hybrid_model_1 import (
    HybridModel1, 
    FoodDataset, 
    val_transform,
    calculate_detailed_metrics
)

warnings.filterwarnings('ignore')

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Apple Silicon (MPS) detected!" if torch.backends.mps.is_available() else "⚠️ CPU detected.")
print(f" Using device: {device}")

class SHAPAnalyzer:
    def __init__(self, model_path, data_dir, num_samples=50):  # Increased from 30
        """
        Initialize SHAP analyzer for hybrid model
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.device = device
        
        # Clear MPS cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Load model
        self.model = self._load_model()
        
        # Prepare data
        self.dataset = self._prepare_dataset()
        
        # Initialize SHAP
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
    def _load_model(self):
        """Load the trained hybrid model"""
        print("Loading model...")
        model = HybridModel1(num_classes=2).to(self.device)
        
        # Load trained weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"✅ Model loaded from {self.model_path}")
        return model
    
    def _prepare_dataset(self):
        """Prepare dataset for SHAP analysis"""
        print("Preparing dataset...")
        
        # Use validation transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FoodDataset(self.data_dir, transform=transform, is_training=False)
        
        # Get balanced samples - increased from 15/15 to 25/25
        good_indices = [i for i, label in enumerate(dataset.labels) if label == 1][:self.num_samples//2]
        bad_indices = [i for i, label in enumerate(dataset.labels) if label == 0][:self.num_samples//2]
        sampled_indices = good_indices + bad_indices
        
        # Create subset
        class SubsetDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
                
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
                
            def __len__(self):
                return len(self.indices)
        
        subset = SubsetDataset(dataset, sampled_indices)
        print(f"✅ Dataset prepared with {len(subset)} samples (balanced: {len(good_indices)} good, {len(bad_indices)} bad)")
        return subset
    
    def _simple_predict(self, x):
        """
        Simple prediction function for SHAP
        """
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Handle different input shapes
        if len(x.shape) == 2:
            # Flattened input from SHAP
            x = x.reshape(-1, 3, 224, 224)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.detach().cpu().numpy()
    
    def compute_shap_values(self):
        """Compute SHAP values with increased samples"""
        print("\n" + "="*60)
        print("Computing SHAP Values (Increased Samples)")
        print("="*60)
        
        # Get data loader with larger batch size
        data_loader = DataLoader(self.dataset, batch_size=16, shuffle=True, num_workers=0)
        
        # Collect more data - aim for 10 background + 10 test
        all_images, all_labels = [], []
        for images, labels in tqdm(data_loader, desc="Loading all data"):
            all_images.append(images)
            all_labels.append(labels)
            if len(all_images) >= 4:  # Limit to 4 batches (64 images max)
                break
        
        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        print(f"Total samples collected: {len(all_images)}")
        
        # Select balanced background samples (10 total: 5 good + 5 bad)
        good_indices = torch.where(all_labels == 1)[0]
        bad_indices = torch.where(all_labels == 0)[0]
        
        # Ensure we have enough samples
        bg_per_class = min(5, len(good_indices), len(bad_indices))
        bg_good = good_indices[:bg_per_class]
        bg_bad = bad_indices[:bg_per_class]
        background_indices = torch.cat([bg_good, bg_bad])
        
        # Select test samples (different from background) - 10 samples
        all_indices = set(range(len(all_images)))
        bg_set = set(background_indices.tolist())
        remaining_indices = list(all_indices - bg_set)
        
        # Select balanced test samples
        remaining_good = [i for i in remaining_indices if all_labels[i] == 1]
        remaining_bad = [i for i in remaining_indices if all_labels[i] == 0]
        
        test_per_class = min(5, len(remaining_good), len(remaining_bad))
        test_indices = remaining_good[:test_per_class] + remaining_bad[:test_per_class]
        
        background = all_images[background_indices]
        test_images = all_images[test_indices]
        test_labels = all_labels[test_indices]
        
        print(f"\n📊 Selected samples:")
        print(f"   Background: {len(background)} samples ({bg_per_class} good, {bg_per_class} bad)")
        print(f"   Test: {len(test_images)} samples ({test_per_class} good, {test_per_class} bad)")
        
        # Flatten images for KernelExplainer
        background_flat = background.numpy().reshape(len(background), -1)
        test_flat = test_images.numpy().reshape(len(test_images), -1)
        
        # Create explainer
        print("\n🔧 Initializing KernelExplainer...")
        self.explainer = shap.KernelExplainer(
            model=self._simple_predict,
            data=background_flat
        )
        
        # Compute SHAP values with progress bar
        print(f"\n⚡ Computing SHAP values for {len(test_flat)} samples...")
        print("   This will take 5-10 minutes. Please be patient...")
        
        all_shap_values = []
        
        # Process samples with tqdm
        for i in tqdm(range(len(test_flat)), desc="Computing SHAP"):
            # Compute SHAP for single sample
            shap_val = self.explainer.shap_values(
                test_flat[i:i+1],
                nsamples=20,  # Increased from 15
                silent=True
            )
            
            # Handle different return formats
            if isinstance(shap_val, list):
                # List of arrays for each class
                # Shape should be [2] where each element is [1, features]
                if len(shap_val) >= 2:
                    all_shap_values.append(shap_val[1][0])  # Class 1, remove batch
                else:
                    all_shap_values.append(shap_val[0][0])
            elif isinstance(shap_val, np.ndarray):
                if len(shap_val.shape) == 3:
                    # Shape: [1, features, classes]
                    all_shap_values.append(shap_val[0])
                else:
                    all_shap_values.append(shap_val)
            else:
                print(f"Unexpected SHAP value type: {type(shap_val)}")
                all_shap_values.append(np.zeros((test_flat.shape[1], 2)))
        
        # Convert to numpy array - shape: [samples, features, classes]
        self.shap_values = np.array(all_shap_values)
        
        print(f"SHAP values shape: {self.shap_values.shape}")
        
        # Get expected value
        background_output = self._simple_predict(background_flat)
        self.expected_value = background_output.mean(axis=0)
        
        print(f"\n✅ SHAP values computed successfully!")
        print(f"   SHAP shape: {self.shap_values.shape}")
        print(f"   Expected value: {self.expected_value}")
        
        return test_images, test_labels
    
    def calculate_shap_metrics(self):
        """Calculate SHAP metrics including mean absolute SHAP"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Run compute_shap_values() first.")
        
        print("\n" + "="*60)
        print("SHAP Metrics Analysis")
        print("="*60)
        
        shap_array = self.shap_values
        print(f"SHAP array shape: {shap_array.shape}")
        
        # Extract class 1 (good quality) SHAP values
        # Assuming shape: [samples, features, classes]
        if len(shap_array.shape) == 3:
            shap_class1 = shap_array[:, :, 1]  # Shape: [samples, features]
        elif len(shap_array.shape) == 2:
            # Already 2D, assume it's class 1
            shap_class1 = shap_array
        else:
            raise ValueError(f"Unexpected SHAP array shape: {shap_array.shape}")
        
        print(f"Class 1 SHAP shape: {shap_class1.shape}")
        
        # Calculate metrics
        mean_abs_shap = np.mean(np.abs(shap_class1), axis=0)  # Shape: [features]
        mean_abs_shap_global = np.mean(np.abs(shap_class1))
        mean_shap = np.mean(shap_class1, axis=0)  # Shape: [features]
        
        # Get top features
        feature_importance = np.argsort(-mean_abs_shap)
        
        # Count positive and negative features
        positive_features = np.sum(mean_shap > 0)
        negative_features = np.sum(mean_shap < 0)
        
        # Create metrics
        metrics = {
            'mean_absolute_shap_global': float(mean_abs_shap_global),
            'mean_absolute_shap_per_feature': mean_abs_shap.tolist(),
            'mean_shap_per_feature': mean_shap.tolist(),
            'top_features': [int(idx) for idx in feature_importance[:100]],
            'expected_value': [float(x) for x in self.expected_value],
            'total_features': int(len(mean_abs_shap)),
            'positive_features': int(positive_features),
            'negative_features': int(negative_features)
        }
        
        # Print summary
        print(f"\n📊 SHAP Metrics Summary:")
        print(f"   Global Mean Absolute SHAP: {mean_abs_shap_global:.6f}")
        print(f"   Expected Value: {self.expected_value}")
        print(f"   Total Features: {len(mean_abs_shap)}")
        print(f"   Positive Impact Features: {positive_features}")
        print(f"   Negative Impact Features: {negative_features}")
        
        # Print top features
        print(f"\n📈 Top 10 Most Important Features:")
        for i in range(min(10, len(feature_importance))):
            feat_idx = int(feature_importance[i])
            mean_abs = float(mean_abs_shap[feat_idx])
            mean_val = float(mean_shap[feat_idx])
            
            # Convert pixel index to approximate position
            pixel_idx = feat_idx
            channel = pixel_idx // (224 * 224)
            pixel_in_channel = pixel_idx % (224 * 224)
            row = pixel_in_channel // 224
            col = pixel_in_channel % 224
            
            impact = "Positive" if mean_val > 0 else "Negative"
            print(f"   {i+1:2d}. Pixel [{row:3d}, {col:3d}] (Channel {channel}):")
            print(f"       Mean |SHAP| = {mean_abs:.6f}")
            print(f"       Mean SHAP = {mean_val:+.6f}")
            print(f"       Impact: {impact}")
        
        return metrics
    
    def create_summary_plot(self, test_images, save_path='shap_summary_plot.png'):
        """Create SHAP summary plot"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")
        
        print(f"\n📈 Creating SHAP Summary Plot...")
        
        # Get SHAP values for class 1
        if len(self.shap_values.shape) == 3:
            shap_class1 = self.shap_values[:, :, 1]  # Shape: [samples, features]
        else:
            shap_class1 = self.shap_values
        
        # Flatten test images
        test_flat = test_images.numpy().reshape(len(test_images), -1)
        
        # Get top 500 most important features for visualization
        mean_abs = np.mean(np.abs(shap_class1), axis=0)
        top_indices = np.argsort(-mean_abs)[:500]
        
        shap_sampled = shap_class1[:, top_indices]
        test_sampled = test_flat[:, top_indices]
        
        # Create feature names
        feature_names = []
        for idx in top_indices:
            pixel_idx = idx
            channel = pixel_idx // (224 * 224)
            pixel_in_channel = pixel_idx % (224 * 224)
            row = pixel_in_channel // 224
            col = pixel_in_channel % 224
            feature_names.append(f"[{row},{col}]_Ch{channel}")
        
        try:
            # Create the summary plot
            plt.figure(figsize=(16, 10))
            
            shap.summary_plot(
                shap_sampled,
                features=test_sampled,
                feature_names=feature_names,
                plot_type="dot",
                show=False,
                max_display=25
            )
            
            plt.title("SHAP Summary Plot - Top 500 Most Important Pixels", 
                     fontsize=18, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Summary plot saved to {save_path}")
            
            # Also create a bar plot version
            plt.figure(figsize=(14, 8))
            shap.summary_plot(
                shap_sampled,
                features=test_sampled,
                plot_type="bar",
                show=False,
                max_display=20
            )
            
            plt.title("SHAP Feature Importance (Bar Plot)", fontsize=18, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('shap_summary_bar.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Summary bar plot saved to shap_summary_bar.png")
            
        except Exception as e:
            print(f"⚠️ Could not create standard summary plot: {e}")
            # Create alternative visualization
            self._create_feature_importance_bar_plot(shap_class1, 'shap_feature_importance.png')
        
        return save_path
    
    def _create_feature_importance_bar_plot(self, shap_class1, save_path):
        """Create bar plot of feature importance"""
        mean_abs = np.mean(np.abs(shap_class1), axis=0)
        top_indices = np.argsort(-mean_abs)[:30]
        top_values = mean_abs[top_indices]
        
        plt.figure(figsize=(14, 10))
        bars = plt.barh(range(30), top_values[::-1], color='steelblue', alpha=0.8)
        
        # Create labels with pixel coordinates
        y_labels = []
        for idx in top_indices[::-1]:
            pixel_idx = idx
            channel = pixel_idx // (224 * 224)
            pixel_in_channel = pixel_idx % (224 * 224)
            row = pixel_in_channel // 224
            col = pixel_in_channel % 224
            y_labels.append(f"[{row:3d},{col:3d}] Ch{channel}")
        
        plt.yticks(range(30), y_labels, fontsize=10)
        plt.xlabel('Mean Absolute SHAP Value', fontsize=14, fontweight='bold')
        plt.title('Top 30 Most Important Pixels', fontsize=18, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                    f'{top_values[len(top_values)-1-i]:.6f}',
                    va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Feature importance bar plot saved to {save_path}")
    
    def create_force_plots(self, test_images, test_labels, num_samples=5):
        """Create SHAP force plots for individual samples - WORKING VERSION"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")
        
        print(f"\n🎨 Creating SHAP Force Plots for {num_samples} samples...")
        
        # Create directory for force plots
        os.makedirs('force_plots', exist_ok=True)
        num_samples = min(num_samples, len(test_images))
        
        # Get predictions
        test_images_device = test_images.to(self.device)
        with torch.no_grad():
            outputs = self.model(test_images_device)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        # Get SHAP values for class 1
        if len(self.shap_values.shape) == 3:
            shap_class1 = self.shap_values[:, :, 1]  # Shape: [samples, features]
        else:
            shap_class1 = self.shap_values
        
        # Create individual force plots
        for i in range(num_samples):
            print(f"\n   Processing sample {i+1}/{num_samples}...")
            
            true_label = test_labels[i].item()
            pred_label = preds[i].item()
            pred_prob = probs[i][1].item()  # Probability of "good"
            
            # Get SHAP values for this sample
            sample_shap = shap_class1[i]
            
            # Get test image flattened
            test_image_flat = test_images[i].numpy().flatten()
            
            # METHOD 1: Create enhanced static force plot
            self._create_enhanced_force_plot(i, sample_shap, test_image_flat, 
                                            true_label, pred_label, pred_prob)
            
            # METHOD 2: Create interactive HTML force plot
            self._create_interactive_force_plot(i, sample_shap, test_image_flat,
                                               true_label, pred_label, pred_prob)
        
        print(f"\n✅ Force plots created in 'force_plots/' directory")
        
        # Create a comprehensive HTML report
        self._create_comprehensive_force_plot_report(test_images, test_labels, num_samples)
    
    def _create_enhanced_force_plot(self, sample_idx, sample_shap, test_image_flat, 
                                   true_label, pred_label, pred_prob):
        """Create enhanced static force plot with better visualization"""
        try:
            # Get top 15 features by absolute SHAP value
            top_n = 15
            top_indices = np.argsort(-np.abs(sample_shap))[:top_n]
            top_shap_values = sample_shap[top_indices]
            
            # Sort by SHAP value for better visualization
            sort_idx = np.argsort(top_shap_values)[::-1]  # Descending
            top_indices = top_indices[sort_idx]
            top_shap_values = top_shap_values[sort_idx]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Subplot 1: Bar plot of top SHAP values
            colors = ['#2E8B57' if v > 0 else '#DC143C' for v in top_shap_values]  # Green/Red
            bars = ax1.bar(range(top_n), top_shap_values, color=colors, alpha=0.8, edgecolor='black')
            
            # Add labels for top features
            x_labels = []
            for idx in top_indices:
                pixel_idx = idx
                channel = pixel_idx // (224 * 224)
                pixel_in_channel = pixel_idx % (224 * 224)
                row = pixel_in_channel // 224
                col = pixel_in_channel % 224
                x_labels.append(f"[{row},{col}]")
            
            ax1.set_xticks(range(top_n))
            ax1.set_xticklabels(x_labels, rotation=45, fontsize=10, fontweight='bold')
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax1.set_ylabel('SHAP Value', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Top Pixel Coordinates', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for j, (bar, val) in enumerate(zip(bars, top_shap_values)):
                height = bar.get_height()
                label_x = bar.get_x() + bar.get_width() / 2
                label_y = height + (0.01 if height >= 0 else -0.01)
                va = 'bottom' if height >= 0 else 'top'
                ax1.text(label_x, label_y, f'{val:.4f}', 
                        ha='center', va=va, fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Subplot 2: Cumulative effect visualization
            # Sort by absolute value for cumulative plot
            abs_sort_idx = np.argsort(np.abs(top_shap_values))[::-1]
            cumulative_values = np.cumsum(top_shap_values[abs_sort_idx])
            
            ax2.plot(range(1, top_n + 1), cumulative_values, 'o-', linewidth=3, 
                    color='darkblue', markersize=8, markerfacecolor='lightblue')
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_xlabel('Number of Top Features', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Cumulative SHAP Value', fontsize=12, fontweight='bold')
            ax2.set_title('Cumulative Feature Impact', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add final prediction info
            final_impact = cumulative_values[-1]
            baseline = self.expected_value[1]
            predicted_prob = baseline + final_impact
            
            ax2.annotate(f'Baseline: {baseline:.3f}\n'
                        f'Final Impact: {final_impact:+.3f}\n'
                        f'Predicted: {predicted_prob:.3f}',
                        xy=(0.7, 0.1), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
                        fontsize=10, fontweight='bold')
            
            # Main title
            title = f'SHAP Force Plot - Sample {sample_idx}\n'
            title += f'True: {"GOOD" if true_label == 1 else "BAD"} | '
            title += f'Predicted: {"GOOD" if pred_label == 1 else "BAD"} | '
            title += f'Probability GOOD: {pred_prob:.3f}'
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            
            # Save static plot
            static_path = f'force_plots/sample_{sample_idx}_force_enhanced.png'
            plt.savefig(static_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      Enhanced force plot saved: {static_path}")
            
        except Exception as e:
            print(f"      Error creating enhanced force plot: {e}")
            # Create simple fallback plot
            self._create_simple_force_plot(sample_idx, sample_shap, true_label, pred_label, pred_prob)
    
    def _create_simple_force_plot(self, sample_idx, sample_shap, true_label, pred_label, pred_prob):
        """Create simple force plot as fallback"""
        try:
            # Get top 10 features
            top_n = 10
            top_indices = np.argsort(-np.abs(sample_shap))[:top_n]
            top_shap_values = sample_shap[top_indices]
            
            plt.figure(figsize=(12, 6))
            
            colors = ['green' if v > 0 else 'red' for v in top_shap_values]
            bars = plt.bar(range(top_n), top_shap_values, color=colors, alpha=0.7)
            
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.xlabel('Top Features', fontsize=12)
            plt.ylabel('SHAP Value', fontsize=12)
            
            # Add title
            title = f'Sample {sample_idx}: '
            title += f'True={"GOOD" if true_label == 1 else "BAD"}, '
            title += f'Pred={"GOOD" if pred_label == 1 else "BAD"} (Prob: {pred_prob:.3f})'
            plt.title(title, fontsize=14, fontweight='bold')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            save_path = f'force_plots/sample_{sample_idx}_force_simple.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      Simple force plot saved: {save_path}")
            
        except Exception as e:
            print(f"      Error creating simple force plot: {e}")
    
    def _create_interactive_force_plot(self, sample_idx, sample_shap, test_image_flat,
                                      true_label, pred_label, pred_prob):
        """Create interactive HTML force plot"""
        try:
            # Limit features for HTML plot to avoid crashes
            max_features_for_html = 500
            if len(sample_shap) > max_features_for_html:
                # Take top features for HTML
                top_idx_html = np.argsort(-np.abs(sample_shap))[:max_features_for_html]
                sample_shap_html = sample_shap[top_idx_html]
                test_image_html = test_image_flat[top_idx_html]
                feature_names = [f'pixel_{j}' for j in range(len(sample_shap_html))]
            else:
                sample_shap_html = sample_shap
                test_image_html = test_image_flat
                feature_names = [f'pixel_{j}' for j in range(len(sample_shap))]
            
            # Create interactive force plot
            force_plot = shap.force_plot(
                base_value=self.expected_value[1],
                shap_values=sample_shap_html,
                features=test_image_html,
                feature_names=feature_names,
                matplotlib=False,
                show=False
            )
            
            # Save as HTML with custom styling
            html_path = f'force_plots/sample_{sample_idx}_force_interactive.html'
            
            # Get the HTML and add custom styling
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>SHAP Force Plot - Sample {sample_idx}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        border-radius: 8px;
                        margin-bottom: 20px;
                    }}
                    .info-box {{
                        background: #e8f4f8;
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 20px;
                    }}
                    .shap-plot {{
                        margin: 20px 0;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        overflow: hidden;
                    }}
                    .feature-summary {{
                        background: #f9f9f9;
                        padding: 15px;
                        border-radius: 8px;
                        margin-top: 20px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 10px;
                    }}
                    th, td {{
                        padding: 10px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }}
                    th {{
                        background-color: #f2f2f2;
                        font-weight: bold;
                    }}
                    .good {{
                        color: green;
                        font-weight: bold;
                    }}
                    .bad {{
                        color: red;
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>SHAP Force Plot Analysis</h1>
                        <h2>Sample {sample_idx}</h2>
                    </div>
                    
                    <div class="info-box">
                        <h3>Prediction Information</h3>
                        <p><strong>True Label:</strong> <span class="{'good' if true_label == 1 else 'bad'}">
                            {"GOOD" if true_label == 1 else "BAD"}</span></p>
                        <p><strong>Predicted Label:</strong> <span class="{'good' if pred_label == 1 else 'bad'}">
                            {"GOOD" if pred_label == 1 else "BAD"}</span></p>
                        <p><strong>Probability (GOOD):</strong> {pred_prob:.3f}</p>
                        <p><strong>Expected Value (Baseline):</strong> {self.expected_value[1]:.3f}</p>
                    </div>
                    
                    <div class="shap-plot">
                        <h3>Interactive SHAP Force Plot</h3>
                        {force_plot.html() if hasattr(force_plot, 'html') else str(force_plot)}
                    </div>
                    
                    <div class="feature-summary">
                        <h3>Top 10 Most Influential Features</h3>
                        <table>
                            <tr>
                                <th>Rank</th>
                                <th>Pixel Coordinates</th>
                                <th>Channel</th>
                                <th>SHAP Value</th>
                                <th>Impact</th>
                            </tr>
            """
            
            # Add top 10 features to table
            top_n = 10
            top_indices = np.argsort(-np.abs(sample_shap))[:top_n]
            for rank, idx in enumerate(top_indices, 1):
                pixel_idx = idx
                channel = pixel_idx // (224 * 224)
                pixel_in_channel = pixel_idx % (224 * 224)
                row = pixel_in_channel // 224
                col = pixel_in_channel % 224
                shap_val = sample_shap[idx]
                impact = "Positive" if shap_val > 0 else "Negative"
                impact_class = "good" if shap_val > 0 else "bad"
                
                html_content += f"""
                            <tr>
                                <td>{rank}</td>
                                <td>[{row}, {col}]</td>
                                <td>{channel}</td>
                                <td>{shap_val:.6f}</td>
                                <td class="{impact_class}">{impact}</td>
                            </tr>
                """
            
            html_content += """
                        </table>
                    </div>
                    
                    <div style="margin-top: 20px; padding: 15px; background: #f0f8ff; border-radius: 8px;">
                        <h4>How to interpret this plot:</h4>
                        <ul>
                            <li><strong>Red bars</strong>: Features that push the prediction toward BAD quality</li>
                            <li><strong>Blue bars</strong>: Features that push the prediction toward GOOD quality</li>
                            <li>The <strong>base value</strong> is the model's output without any features</li>
                            <li>The <strong>output value</strong> is the final prediction after adding all feature contributions</li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """
            
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            print(f"      Interactive force plot saved: {html_path}")
            
        except Exception as e:
            print(f"      Could not create interactive plot: {e}")
    
    def _create_comprehensive_force_plot_report(self, test_images, test_labels, num_samples):
        """Create a comprehensive HTML report with all force plots"""
        try:
            # Get predictions
            test_images_device = test_images.to(self.device)
            with torch.no_grad():
                outputs = self.model(test_images_device)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
            
            # Get SHAP values for class 1
            if len(self.shap_values.shape) == 3:
                shap_class1 = self.shap_values[:, :, 1]
            else:
                shap_class1 = self.shap_values
            
            # Create HTML content
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Comprehensive SHAP Force Plots Analysis</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                    .container { max-width: 1400px; margin: 0 auto; }
                    .header { 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white; 
                        padding: 30px; 
                        border-radius: 10px;
                        margin-bottom: 30px;
                        text-align: center;
                    }
                    .sample-card {
                        background: white;
                        border-radius: 10px;
                        padding: 20px;
                        margin: 20px 0;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        transition: transform 0.3s;
                    }
                    .sample-card:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
                    }
                    .sample-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding-bottom: 15px;
                        border-bottom: 2px solid #eee;
                        margin-bottom: 15px;
                    }
                    .prediction-info {
                        background: #f8f9fa;
                        padding: 15px;
                        border-radius: 8px;
                        margin: 15px 0;
                    }
                    .plot-container {
                        margin: 20px 0;
                        text-align: center;
                    }
                    .plot-container img {
                        max-width: 100%;
                        height: auto;
                        border-radius: 8px;
                        border: 1px solid #ddd;
                    }
                    .good { color: #28a745; font-weight: bold; }
                    .bad { color: #dc3545; font-weight: bold; }
                    .correct { background: #d4edda; }
                    .incorrect { background: #f8d7da; }
                    .stats-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 15px;
                        margin: 20px 0;
                    }
                    .stat-card {
                        background: white;
                        padding: 15px;
                        border-radius: 8px;
                        text-align: center;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }
                    .stat-value {
                        font-size: 24px;
                        font-weight: bold;
                        margin: 10px 0;
                    }
                    .feature-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 15px 0;
                    }
                    .feature-table th, .feature-table td {
                        padding: 10px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }
                    .feature-table th {
                        background: #f8f9fa;
                        font-weight: bold;
                    }
                    .toggle-btn {
                        background: #007bff;
                        color: white;
                        border: none;
                        padding: 8px 15px;
                        border-radius: 5px;
                        cursor: pointer;
                        margin: 10px 0;
                    }
                    .toggle-btn:hover {
                        background: #0056b3;
                    }
                    .hidden-content {
                        display: none;
                        padding: 15px;
                        background: #f8f9fa;
                        border-radius: 8px;
                        margin-top: 10px;
                    }
                </style>
                <script>
                    function toggleFeatures(sampleId) {
                        var content = document.getElementById('features-' + sampleId);
                        var btn = document.getElementById('toggle-btn-' + sampleId);
                        if (content.style.display === 'none') {
                            content.style.display = 'block';
                            btn.textContent = 'Hide Top Features';
                        } else {
                            content.style.display = 'none';
                            btn.textContent = 'Show Top 10 Features';
                        }
                    }
                    
                    function toggleAllFeatures() {
                        var allHidden = document.querySelectorAll('[id^="features-"]');
                        var allButtons = document.querySelectorAll('[id^="toggle-btn-"]');
                        var shouldShow = allHidden[0].style.display === 'none';
                        
                        allHidden.forEach(function(el) {
                            el.style.display = shouldShow ? 'block' : 'none';
                        });
                        
                        allButtons.forEach(function(btn) {
                            btn.textContent = shouldShow ? 'Hide Top Features' : 'Show Top 10 Features';
                        });
                    }
                </script>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Comprehensive SHAP Force Plots Analysis</h1>
                        <h2>Hybrid Model: EfficientNetV2M + MobileViT</h2>
                        <p>Food Quality Classification - Detailed Prediction Explanations</p>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-label">Total Samples</div>
                            <div class="stat-value">""" + str(num_samples) + """</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Expected Value</div>
                            <div class="stat-value">""" + f"{self.expected_value[1]:.3f}" + """</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Mean |SHAP|</div>
                            <div class="stat-value">""" + f"{np.mean(np.abs(shap_class1)):.6f}" + """</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Analysis Date</div>
                            <div class="stat-value">""" + pd.Timestamp.now().strftime('%m/%d') + """</div>
                        </div>
                    </div>
                    
                    <div style="text-align: center; margin: 20px 0;">
                        <button class="toggle-btn" onclick="toggleAllFeatures()">Show/Hide All Feature Tables</button>
                    </div>
            """
            
            # Add each sample
            for i in range(min(num_samples, len(test_images))):
                true_label = test_labels[i].item()
                pred_label = preds[i].item()
                pred_prob = probs[i][1].item()
                correct = true_label == pred_label
                
                # Get top 10 features for this sample
                sample_shap = shap_class1[i]
                top_indices = np.argsort(-np.abs(sample_shap))[:10]
                
                html_content += f"""
                <div class="sample-card {'correct' if correct else 'incorrect'}">
                    <div class="sample-header">
                        <h2>Sample {i}</h2>
                        <div>
                            <span class="{'good' if true_label == 1 else 'bad'}">True: {"GOOD" if true_label == 1 else "BAD"}</span> | 
                            <span class="{'good' if pred_label == 1 else 'bad'}">Pred: {"GOOD" if pred_label == 1 else "BAD"}</span> | 
                            <span>Prob: {pred_prob:.3f}</span> | 
                            <span class="{'good' if correct else 'bad'}">{'✓ Correct' if correct else '✗ Incorrect'}</span>
                        </div>
                    </div>
                    
                    <div class="prediction-info">
                        <p><strong>Prediction Analysis:</strong> This sample was classified as <span class="{'good' if pred_label == 1 else 'bad'}">{"GOOD" if pred_label == 1 else "BAD"}</span> 
                        with {pred_prob:.1%} confidence. The model's baseline expectation was {self.expected_value[1]:.3f}.</p>
                    </div>
                    
                    <div class="plot-container">
                        <h3>Enhanced Force Plot</h3>
                        <img src="sample_{i}_force_enhanced.png" alt="Force Plot Sample {i}">
                        <p><a href="sample_{i}_force_interactive.html" target="_blank">Open Interactive Version</a></p>
                    </div>
                    
                    <button id="toggle-btn-{i}" class="toggle-btn" onclick="toggleFeatures({i})">Show Top 10 Features</button>
                    
                    <div id="features-{i}" class="hidden-content">
                        <h4>Top 10 Most Influential Pixels:</h4>
                        <table class="feature-table">
                            <tr>
                                <th>Rank</th>
                                <th>Coordinates</th>
                                <th>Channel</th>
                                <th>SHAP Value</th>
                                <th>Impact</th>
                                <th>Description</th>
                            </tr>
                """
                
                # Add feature rows
                for rank, idx in enumerate(top_indices, 1):
                    pixel_idx = idx
                    channel = pixel_idx // (224 * 224)
                    pixel_in_channel = pixel_idx % (224 * 224)
                    row = pixel_in_channel // 224
                    col = pixel_in_channel % 224
                    shap_val = sample_shap[idx]
                    impact = "Positive" if shap_val > 0 else "Negative"
                    impact_class = "good" if shap_val > 0 else "bad"
                    
                    # Add some descriptive text based on position and impact
                    if channel == 0:
                        channel_desc = "Red"
                    elif channel == 1:
                        channel_desc = "Green"
                    else:
                        channel_desc = "Blue"
                    
                    description = f"{impact} impact from {channel_desc.lower()} channel"
                    if row < 50 and col < 50:
                        description += " in top-left corner"
                    elif row < 50 and col > 174:
                        description += " in top-right corner"
                    elif row > 174 and col < 50:
                        description += " in bottom-left corner"
                    elif row > 174 and col > 174:
                        description += " in bottom-right corner"
                    else:
                        description += " in central region"
                    
                    html_content += f"""
                            <tr>
                                <td>{rank}</td>
                                <td>[{row}, {col}]</td>
                                <td>{channel_desc} ({channel})</td>
                                <td>{shap_val:.6f}</td>
                                <td class="{impact_class}">{impact}</td>
                                <td>{description}</td>
                            </tr>
                    """
                
                html_content += f"""
                        </table>
                    </div>
                </div>
                """
            
            html_content += """
                </div>
            </body>
            </html>
            """
            
            # Save HTML file
            with open('force_plots/comprehensive_report.html', 'w') as f:
                f.write(html_content)
            
            print(f"\n📊 Comprehensive report saved: force_plots/comprehensive_report.html")
            print("   Open this file in your browser to view all force plots interactively!")
            
        except Exception as e:
            print(f"⚠️ Could not create comprehensive report: {e}")
    
    def create_heatmap_plots(self, test_images, num_samples=3):
        """Create SHAP heatmap plots"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")
        
        print(f"\n🔥 Creating SHAP Heatmap Plots for {num_samples} samples...")
        
        os.makedirs('heatmap_plots', exist_ok=True)
        num_samples = min(num_samples, len(test_images))
        
        # Get SHAP values for class 1
        if len(self.shap_values.shape) == 3:
            shap_class1 = self.shap_values[:, :, 1]
        else:
            shap_class1 = self.shap_values
        
        for i in range(num_samples):
            print(f"   Processing heatmap for sample {i}...")
            
            # Get original image (denormalized)
            img = test_images[i].numpy()
            img = self._denormalize_image(img)
            
            # Get SHAP values for this sample and reshape to image
            sample_shap = shap_class1[i].reshape(3, 224, 224)
            
            # Calculate importance per channel and overall
            shap_abs = np.abs(sample_shap)
            shap_sum = np.sum(shap_abs, axis=0)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Original image
            axes[0, 0].imshow(img)
            axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Overall SHAP heatmap
            im1 = axes[0, 1].imshow(shap_sum, cmap='hot')
            axes[0, 1].set_title('Overall SHAP Importance', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            # Overlay
            axes[0, 2].imshow(img, alpha=0.7)
            overlay1 = axes[0, 2].imshow(shap_sum, cmap='hot', alpha=0.5)
            axes[0, 2].set_title('SHAP Overlay', fontsize=14, fontweight='bold')
            axes[0, 2].axis('off')
            plt.colorbar(overlay1, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # Channel-specific heatmaps
            channels = ['Red', 'Green', 'Blue']
            channel_colors = ['Reds', 'Greens', 'Blues']
            for ch in range(3):
                row = 1
                col = ch
                im = axes[row, col].imshow(shap_abs[ch], cmap=channel_colors[ch])
                axes[row, col].set_title(f'{channels[ch]} Channel SHAP', fontsize=12, fontweight='bold')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            
            plt.suptitle(f'Sample {i} - SHAP Heatmap Analysis', fontsize=18, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            save_path = f'heatmap_plots/sample_{i}_heatmap.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      Saved heatmap plot: {save_path}")
        
        print(f"\n✅ Heatmap plots created in 'heatmap_plots/' directory")
    
    def _denormalize_image(self, image):
        """Denormalize image for display"""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image = image.transpose(1, 2, 0)
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        return image
    
    def generate_comprehensive_report(self, metrics, test_images, test_labels):
        """Generate comprehensive SHAP analysis report"""
        print("\n" + "="*60)
        print("Generating Comprehensive SHAP Analysis Report")
        print("="*60)
        
        # Get model predictions
        test_images_device = test_images.to(self.device)
        with torch.no_grad():
            outputs = self.model(test_images_device)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        accuracy = (preds.cpu().numpy() == test_labels.numpy()).mean()
        
        # Calculate confusion matrix
        cm = confusion_matrix(test_labels.numpy(), preds.cpu().numpy())
        
        # Create comprehensive report
        report = {
            'analysis_metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'shap_version': shap.__version__,
                'device': str(self.device),
                'model': 'HybridModel1 (EfficientNetV2M + MobileViT)',
                'samples_analyzed': len(test_images),
                'background_samples': 10,  # Increased from 4
                'test_samples': len(test_images)
            },
            'performance_metrics': {
                'accuracy': float(accuracy),
                'confusion_matrix': cm.tolist(),
                'class_distribution': {
                    'good_samples': int(np.sum(test_labels.numpy() == 1)),
                    'bad_samples': int(np.sum(test_labels.numpy() == 0))
                }
            },
            'shap_analysis_results': {
                'global_metrics': {
                    'mean_absolute_shap': float(metrics['mean_absolute_shap_global']),
                    'expected_value': metrics['expected_value']
                },
                'feature_analysis': {
                    'total_features': metrics['total_features'],
                    'positive_impact_features': metrics['positive_features'],
                    'negative_impact_features': metrics['negative_features'],
                    'top_10_features': []
                }
            },
            'visualizations_generated': {
                'summary_plots': [
                    'shap_summary_plot.png',
                    'shap_summary_bar.png',
                    'shap_feature_importance.png'
                ],
                'force_plots': {
                    'directory': 'force_plots/',
                    'files': [
                        'comprehensive_report.html (main report)',
                        'sample_*_force_enhanced.png (static plots)',
                        'sample_*_force_interactive.html (interactive plots)'
                    ]
                },
                'heatmap_plots': {
                    'directory': 'heatmap_plots/',
                    'files': 'sample_*_heatmap.png'
                }
            },
            'key_findings': {
                'model_interpretability': 'SHAP analysis reveals which image pixels influence predictions',
                'feature_importance': 'Top influential pixels identified with spatial coordinates',
                'prediction_explanations': 'Force plots show how individual features affect each prediction'
            }
        }
        
        # Add top features with details
        top_indices = metrics['top_features'][:10]
        for i, idx in enumerate(top_indices):
            mean_abs = float(metrics['mean_absolute_shap_per_feature'][idx])
            mean_shap = float(metrics['mean_shap_per_feature'][idx])
            
            pixel_idx = idx
            channel = pixel_idx // (224 * 224)
            pixel_in_channel = pixel_idx % (224 * 224)
            row = pixel_in_channel // 224
            col = pixel_in_channel % 224
            
            report['shap_analysis_results']['feature_analysis']['top_10_features'].append({
                'rank': i + 1,
                'pixel_index': pixel_idx,
                'coordinates': {'row': int(row), 'col': int(col), 'channel': int(channel)},
                'shap_values': {
                    'mean_absolute': mean_abs,
                    'mean': mean_shap,
                    'impact': 'positive' if mean_shap > 0 else 'negative'
                }
            })
        
        # Save JSON report
        with open('shap_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print executive summary
        print("\n" + "="*60)
        print("EXECUTIVE SUMMARY")
        print("="*60)
        
        print(f"\n📊 ANALYSIS SCOPE:")
        print(f"   Samples Analyzed: {len(test_images)} (increased from 4 to 10)")
        print(f"   Background Samples: 10 (5 GOOD + 5 BAD)")
        print(f"   Test Samples: {len(test_images)} ({report['performance_metrics']['class_distribution']['good_samples']} GOOD, "
              f"{report['performance_metrics']['class_distribution']['bad_samples']} BAD)")
        
        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Confusion Matrix:")
        print(f"                Predicted")
        print(f"               GOOD   BAD")
        print(f"   Actual GOOD   {cm[1, 1]:3d}    {cm[1, 0]:3d}")
        print(f"   Actual BAD    {cm[0, 1]:3d}    {cm[0, 0]:3d}")
        
        print(f"\n📈 SHAP ANALYSIS RESULTS:")
        print(f"   Mean Absolute SHAP: {metrics['mean_absolute_shap_global']:.6f}")
        print(f"   Expected Value: GOOD={metrics['expected_value'][1]:.3f}, BAD={metrics['expected_value'][0]:.3f}")
        print(f"   Positive Impact Features: {metrics['positive_features']}")
        print(f"   Negative Impact Features: {metrics['negative_features']}")
        
        print(f"\n🎨 VISUALIZATIONS CREATED:")
        print(f"   1. Summary Plots:")
        print(f"      • shap_summary_plot.png - Top 500 important pixels")
        print(f"      • shap_summary_bar.png - Feature importance bar chart")
        print(f"      • shap_feature_importance.png - Top 30 pixels with coordinates")
        
        print(f"\n   2. Force Plots (PRIMARY OUTPUT):")
        print(f"      • force_plots/comprehensive_report.html - MAIN REPORT")
        print(f"        Open this in browser for interactive analysis!")
        print(f"      • force_plots/sample_*_force_enhanced.png - Static plots")
        print(f"      • force_plots/sample_*_force_interactive.html - Interactive plots")
        
        print(f"\n   3. Heatmap Plots:")
        print(f"      • heatmap_plots/sample_*_heatmap.png - Spatial importance maps")
        
        print(f"\n📁 FILES GENERATED:")
        print(f"   • shap_analysis_report.json - Complete metrics and analysis")
        print(f"   • All visualizations in their respective directories")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Open 'force_plots/comprehensive_report.html' in your web browser")
        print(f"   2. Review the interactive force plots for each sample")
        print(f"   3. Check which pixels influence GOOD vs BAD predictions")
        print(f"   4. Use insights to understand model decision-making")
        
        print(f"\n✅ Analysis complete! Increased sample size provides more reliable insights.")
        
        return report

def main():
    """Main execution function"""
    print("🎯 COMPLETE SHAP ANALYSIS WITH FORCE PLOTS & INCREASED SAMPLES")
    print("="*70)
    print(f"SHAP Version: {shap.__version__}")
    print("="*70)
    
    # Configuration - INCREASED SAMPLE SIZE
    MODEL_PATH = "best_hybrid_model1.pth"
    DATA_DIR = "/Users/tanaypatel/Documents/projects/Machine_Learning/KFU/food_classification_project/Master Dataset"
    
    try:
        # Initialize analyzer with INCREASED samples (50 total, 25/25 balanced)
        print("\n" + "="*60)
        print("INITIALIZING ANALYZER WITH INCREASED SAMPLES")
        print("="*60)
        
        analyzer = SHAPAnalyzer(
            model_path=MODEL_PATH,
            data_dir=DATA_DIR,
            num_samples=50  # Increased from 30 to 50
        )
        
        # Compute SHAP values with increased background/test samples
        print("\n" + "="*60)
        print("STEP 1: COMPUTING SHAP VALUES (10 background + 10 test)")
        print("="*60)
        test_images, test_labels = analyzer.compute_shap_values()
        
        # Calculate metrics
        print("\n" + "="*60)
        print("STEP 2: CALCULATING SHAP METRICS")
        print("="*60)
        metrics = analyzer.calculate_shap_metrics()
        
        # Create ALL visualizations
        print("\n" + "="*60)
        print("STEP 3: CREATING COMPLETE VISUALIZATION SUITE")
        print("="*60)
        
        # 1. Summary plots (dot and bar)
        print("\n📊 Creating Summary Plots...")
        analyzer.create_summary_plot(test_images)
        
        # 2. Force plots (ENHANCED - now with interactive HTML)
        print("\n🎨 Creating Enhanced Force Plots with Interactive HTML...")
        analyzer.create_force_plots(test_images, test_labels, num_samples=5)
        
        # 3. Heatmap plots
        print("\n🔥 Creating Heatmap Plots...")
        analyzer.create_heatmap_plots(test_images, num_samples=3)
        
        # Generate comprehensive report
        print("\n" + "="*60)
        print("STEP 4: GENERATING FINAL REPORT")
        print("="*60)
        report = analyzer.generate_comprehensive_report(metrics, test_images, test_labels)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE WITH INCREASED SAMPLES!")
        print("="*70)
        
        # Print final summary
        print("\n📋 KEY IMPROVEMENTS IN THIS VERSION:")
        print("1. ✅ Increased samples: 10 background + 10 test (from 4+4)")
        print("2. ✅ Working force plots: Enhanced static + Interactive HTML")
        print("3. ✅ Comprehensive HTML report with all samples")
        print("4. ✅ Better error handling and visualization")
        
        print("\n🔍 WHAT TO DO NEXT:")
        print("1. 📂 Open 'force_plots/comprehensive_report.html' in Chrome/Firefox")
        print("2. 🔍 Examine how specific pixels influence predictions")
        print("3. 📊 Compare GOOD vs BAD sample explanations")
        print("4. 🎯 Identify model biases and important visual features")
        
        print("\n💡 TIPS FOR INTERPRETATION:")
        print("• Green/positive features push toward GOOD classification")
        print("• Red/negative features push toward BAD classification")
        print("• Look for spatial patterns in heatmaps")
        print("• Compare feature importance across different food types")
        
        # Cleanup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 If you encounter memory issues on M3 MacBook Air:")
        print("   • Reduce num_samples from 50 to 40")
        print("   • Close other applications")
        print("   • The code will still work with fewer samples")

if __name__ == "__main__":
    main()
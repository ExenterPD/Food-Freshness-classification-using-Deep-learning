# SHAP Analysis with Proper Force Plots (Like Your Example Image)
# Clean, Minimal Implementation

import torch
import torch.nn as nn
import numpy as np
import shap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
from tqdm import tqdm
import warnings
import matplotlib

# Import your model architecture
sys.path.append('.')
from hybrid_model_1 import HybridModel1, FoodDataset

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Apple Silicon (MPS) detected!" if torch.backends.mps.is_available() else "⚠️ CPU detected.")
print(f" Using device: {device}")

class SHAPForcePlotAnalyzer:
    def __init__(self, model_path, data_dir, num_samples=20):
        """
        Initialize analyzer for clean force plots
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.device = device
        
        # Clear cache
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
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Model loaded from {self.model_path}")
        return model
    
    def _prepare_dataset(self):
        """Prepare dataset for SHAP analysis"""
        print("Preparing dataset...")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FoodDataset(self.data_dir, transform=transform, is_training=False)
        
        # Get balanced samples
        good_indices = [i for i, label in enumerate(dataset.labels) if label == 1][:self.num_samples//2]
        bad_indices = [i for i, label in enumerate(dataset.labels) if label == 0][:self.num_samples//2]
        sampled_indices = good_indices + bad_indices
        
        class SubsetDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
                
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
                
            def __len__(self):
                return len(self.indices)
        
        subset = SubsetDataset(dataset, sampled_indices)
        print(f"✅ Dataset prepared with {len(subset)} samples")
        return subset
    
    def _simple_predict(self, x):
        """
        Simple prediction function for SHAP
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if len(x.shape) == 2:
            x = x.reshape(-1, 3, 224, 224)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.detach().cpu().numpy()
    
    def compute_shap_values(self):
        """Compute SHAP values"""
        print("\n" + "="*60)
        print("Computing SHAP Values")
        print("="*60)
        
        # Get data loader
        data_loader = DataLoader(self.dataset, batch_size=8, shuffle=True)
        
        # Collect data
        all_images, all_labels = [], []
        for images, labels in data_loader:
            all_images.append(images)
            all_labels.append(labels)
            if len(all_images) >= 4:
                break
        
        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        print(f"Total samples collected: {len(all_images)}")
        
        # Select background samples
        good_indices = torch.where(all_labels == 1)[0]
        bad_indices = torch.where(all_labels == 0)[0]
        
        bg_per_class = min(3, len(good_indices), len(bad_indices))
        background_indices = torch.cat([good_indices[:bg_per_class], bad_indices[:bg_per_class]])
        
        # Select test samples
        all_indices = set(range(len(all_images)))
        bg_set = set(background_indices.tolist())
        remaining_indices = list(all_indices - bg_set)
        
        remaining_good = [i for i in remaining_indices if all_labels[i] == 1]
        remaining_bad = [i for i in remaining_indices if all_labels[i] == 0]
        
        test_per_class = min(3, len(remaining_good), len(remaining_bad))
        test_indices = remaining_good[:test_per_class] + remaining_bad[:test_per_class]
        
        background = all_images[background_indices]
        test_images = all_images[test_indices]
        test_labels = all_labels[test_indices]
        
        print(f"\n📊 Selected samples:")
        print(f"   Background: {len(background)} samples")
        print(f"   Test: {len(test_images)} samples")
        
        # Flatten images
        background_flat = background.numpy().reshape(len(background), -1)
        test_flat = test_images.numpy().reshape(len(test_images), -1)
        
        # Create explainer
        print("\n🔧 Initializing KernelExplainer...")
        self.explainer = shap.KernelExplainer(
            model=self._simple_predict,
            data=background_flat
        )
        
        # Compute SHAP values
        print(f"\n⚡ Computing SHAP values...")
        
        all_shap_values = []
        
        for i in tqdm(range(len(test_flat)), desc="Computing SHAP"):
            shap_val = self.explainer.shap_values(
                test_flat[i:i+1],
                nsamples=10,  # Reduced for speed
                silent=True
            )
            
            if isinstance(shap_val, list):
                if len(shap_val) >= 2:
                    all_shap_values.append(shap_val[1][0])
                else:
                    all_shap_values.append(shap_val[0][0])
            elif isinstance(shap_val, np.ndarray):
                if len(shap_val.shape) == 3:
                    all_shap_values.append(shap_val[0])
                else:
                    all_shap_values.append(shap_val)
            else:
                all_shap_values.append(np.zeros((test_flat.shape[1], 2)))
        
        self.shap_values = np.array(all_shap_values)
        
        # Get expected value
        background_output = self._simple_predict(background_flat)
        self.expected_value = background_output.mean(axis=0)
        
        print(f"\n✅ SHAP values computed!")
        print(f"   SHAP shape: {self.shap_values.shape}")
        print(f"   Expected value: {self.expected_value}")
        
        return test_images, test_labels
    
    def create_proper_force_plots(self, test_images, test_labels, num_samples=5):
        """Create proper SHAP force plots like in your example"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")
        
        print(f"\n🎨 Creating Proper SHAP Force Plots for {num_samples} samples...")
        
        # Create directory
        os.makedirs('shap_force_plots', exist_ok=True)
        num_samples = min(num_samples, len(test_images))
        
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
        
        # Create force plots for each sample
        for i in range(num_samples):
            print(f"   Creating force plot for sample {i+1}/{num_samples}...")
            
            true_label = test_labels[i].item()
            pred_label = preds[i].item()
            pred_prob = probs[i][1].item()
            
            # Get SHAP values for this sample
            sample_shap = shap_class1[i]
            
            # Get top features (limit to top 20 for readability)
            top_n = min(20, len(sample_shap))
            top_indices = np.argsort(-np.abs(sample_shap))[:top_n]
            top_shap_values = sample_shap[top_indices]
            
            # Get feature names
            feature_names = []
            for idx in top_indices:
                pixel_idx = idx
                channel = pixel_idx // (224 * 224)
                pixel_in_channel = pixel_idx % (224 * 224)
                row = pixel_in_channel // 224
                col = pixel_in_channel % 224
                feature_names.append(f"[{row},{col}]_Ch{channel}")
            
            # Create the force plot
            self._create_single_force_plot(
                sample_idx=i,
                shap_values=top_shap_values,
                feature_names=feature_names,
                true_label=true_label,
                pred_label=pred_label,
                pred_prob=pred_prob,
                top_indices=top_indices
            )
        
        print(f"\n✅ Force plots saved in 'shap_force_plots/' directory")
    
    def _create_single_force_plot(self, sample_idx, shap_values, feature_names, 
                                 true_label, pred_label, pred_prob, top_indices):
        """Create a single force plot like in your example"""
        try:
            # Sort by SHAP value for better visualization
            sort_idx = np.argsort(shap_values)[::-1]  # Descending
            shap_values_sorted = shap_values[sort_idx]
            feature_names_sorted = [feature_names[i] for i in sort_idx]
            
            # Calculate cumulative sum for positioning
            cumulative = np.cumsum(shap_values_sorted)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # ========== MAIN FORCE PLOT ==========
            baseline = self.expected_value[1]
            final_value = baseline + np.sum(shap_values_sorted)
            
            # Plot baseline
            ax1.axvline(x=baseline, color='black', linestyle='-', linewidth=2, alpha=0.7)
            ax1.text(baseline, -0.5, 'base value', ha='center', va='top', 
                    fontsize=10, fontweight='bold', color='black')
            
            # Plot final value
            ax1.axvline(x=final_value, color='blue', linestyle='-', linewidth=2, alpha=0.7)
            ax1.text(final_value, -0.5, 'output value', ha='center', va='top',
                    fontsize=10, fontweight='bold', color='blue')
            
            # Plot the force arrows
            current_x = baseline
            y_pos = 0
            
            for i, (shap_val, feat_name) in enumerate(zip(shap_values_sorted, feature_names_sorted)):
                # Determine color
                if shap_val > 0:
                    color = '#2E8B57'  # Green for positive
                    arrow_color = 'darkgreen'
                else:
                    color = '#DC143C'  # Red for negative
                    arrow_color = 'darkred'
                
                # Plot the bar
                bar = ax1.barh(y_pos, shap_val, left=current_x, 
                              color=color, alpha=0.7, height=0.6)
                
                # Add feature name
                mid_x = current_x + shap_val / 2
                ax1.text(mid_x, y_pos, feat_name, ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
                
                # Add SHAP value
                value_text = f'{shap_val:+.3f}'
                ax1.text(current_x + shap_val + (0.05 if shap_val > 0 else -0.05), 
                        y_pos, value_text, ha='left' if shap_val > 0 else 'right',
                        va='center', fontsize=9, fontweight='bold', color=arrow_color)
                
                # Update position
                current_x += shap_val
                y_pos += 1
            
            # Set up the main plot
            ax1.set_xlabel('Model Output', fontsize=12, fontweight='bold')
            ax1.set_title(f'Sample {sample_idx} - SHAP Force Plot', 
                         fontsize=14, fontweight='bold', pad=20)
            
            # Add prediction info
            pred_info = f"True: {'GOOD' if true_label == 1 else 'BAD'} | "
            pred_info += f"Pred: {'GOOD' if pred_label == 1 else 'BAD'} | "
            pred_info += f"Prob: {pred_prob:.3f}"
            
            ax1.text(0.02, 1.02, pred_info, transform=ax1.transAxes,
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
            
            ax1.grid(True, alpha=0.3, axis='x')
            ax1.set_ylim(-1, len(shap_values_sorted))
            ax1.set_yticks([])
            
            # ========== FEATURE IMPORTANCE BAR CHART ==========
            # Plot feature importance as bar chart
            y_positions = np.arange(len(feature_names_sorted))
            bars = ax2.barh(y_positions, np.abs(shap_values_sorted), 
                          color=['#2E8B57' if v > 0 else '#DC143C' for v in shap_values_sorted],
                          alpha=0.7)
            
            ax2.set_yticks(y_positions)
            ax2.set_yticklabels(feature_names_sorted, fontsize=9)
            ax2.set_xlabel('|SHAP Value| (Absolute Impact)', fontsize=10)
            ax2.set_title('Feature Importance (Absolute Values)', fontsize=11, fontweight='bold')
            ax2.invert_yaxis()  # Highest importance at top
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add values to bars
            for bar, val in zip(bars, np.abs(shap_values_sorted)):
                width = bar.get_width()
                ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=8, fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            save_path = f'shap_force_plots/sample_{sample_idx}_force.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      Saved: {save_path}")
            
            # Also create a simplified version
            self._create_simplified_force_plot(sample_idx, shap_values_sorted, 
                                             feature_names_sorted, true_label, 
                                             pred_label, pred_prob)
            
        except Exception as e:
            print(f"      Error creating force plot: {e}")
    
    def _create_simplified_force_plot(self, sample_idx, shap_values, feature_names, 
                                     true_label, pred_label, pred_prob):
        """Create a simplified force plot focused on the main visualization"""
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            baseline = self.expected_value[1]
            final_value = baseline + np.sum(shap_values)
            
            # Plot baseline and final value
            ax.axvline(x=baseline, color='black', linestyle='-', linewidth=2, alpha=0.7)
            ax.text(baseline, -0.5, 'base value', ha='center', va='top', 
                   fontsize=10, fontweight='bold', color='black')
            
            ax.axvline(x=final_value, color='blue', linestyle='-', linewidth=2, alpha=0.7)
            ax.text(final_value, -0.5, 'output value', ha='center', va='top',
                   fontsize=10, fontweight='bold', color='blue')
            
            # Plot the forces
            current_x = baseline
            y_pos = 0
            
            # Only show top 10 features for simplicity
            n_features = min(10, len(shap_values))
            
            for i in range(n_features):
                shap_val = shap_values[i]
                feat_name = feature_names[i]
                
                # Determine color
                if shap_val > 0:
                    color = '#2E8B57'  # Green
                else:
                    color = '#DC143C'  # Red
                
                # Plot bar
                ax.barh(y_pos, shap_val, left=current_x, 
                       color=color, alpha=0.8, height=0.5)
                
                # Add feature name (abbreviated)
                short_name = feat_name.replace('_Ch', 'C')[:10]
                mid_x = current_x + shap_val / 2
                ax.text(mid_x, y_pos, short_name, ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
                
                # Update position
                current_x += shap_val
                y_pos += 1
            
            # Add prediction info
            ax.text(0.02, 1.02, 
                   f"Sample {sample_idx} | True: {'GOOD' if true_label == 1 else 'BAD'} | "
                   f"Pred: {'GOOD' if pred_label == 1 else 'BAD'} | Prob: {pred_prob:.3f}",
                   transform=ax.transAxes, fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Model Output Value', fontsize=11, fontweight='bold')
            ax.set_title('SHAP Force Plot', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_ylim(-1, n_features)
            ax.set_yticks([])
            
            plt.tight_layout()
            
            save_path = f'shap_force_plots/sample_{sample_idx}_force_simple.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"      Error creating simplified plot: {e}")
    
    def create_summary_visualizations(self, test_images, test_labels):
        """Create summary visualizations"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed.")
        
        print("\n📊 Creating Summary Visualizations...")
        
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
        
        # Create summary figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Mean absolute SHAP across samples
        mean_abs_per_sample = np.mean(np.abs(shap_class1), axis=1)
        axes[0, 0].bar(range(len(mean_abs_per_sample)), mean_abs_per_sample, 
                      color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Sample Index', fontsize=11)
        axes[0, 0].set_ylabel('Mean |SHAP|', fontsize=11)
        axes[0, 0].set_title('Feature Impact per Sample', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution of SHAP values
        all_shap = shap_class1.flatten()
        all_shap = all_shap[np.abs(all_shap) > 0.0001]
        
        axes[0, 1].hist(all_shap, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 1].axvline(x=0, color='black', linestyle='-', linewidth=2)
        axes[0, 1].set_xlabel('SHAP Value', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Distribution of SHAP Values', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Top features overall
        mean_abs_all = np.mean(np.abs(shap_class1), axis=0)
        top_indices = np.argsort(-mean_abs_all)[:10]
        top_values = mean_abs_all[top_indices]
        
        y_labels = []
        for idx in top_indices:
            pixel_idx = idx
            channel = pixel_idx // (224 * 224)
            pixel_in_channel = pixel_idx % (224 * 224)
            row = pixel_in_channel // 224
            col = pixel_in_channel % 224
            y_labels.append(f"[{row},{col}]C{channel}")
        
        axes[1, 0].barh(range(10), top_values[::-1], color='darkred', alpha=0.7)
        axes[1, 0].set_yticks(range(10))
        axes[1, 0].set_yticklabels(y_labels[::-1], fontsize=9)
        axes[1, 0].set_xlabel('Mean |SHAP|', fontsize=11)
        axes[1, 0].set_title('Top 10 Most Important Pixels', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. Accuracy and predictions
        correct = (preds.cpu().numpy() == test_labels.numpy())
        accuracy = np.mean(correct)
        
        categories = ['Correct', 'Incorrect']
        counts = [np.sum(correct), len(correct) - np.sum(correct)]
        colors = ['#2E8B57', '#DC143C']
        
        axes[1, 1].pie(counts, labels=categories, colors=colors, autopct='%1.1f%%',
                      startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        axes[1, 1].set_title(f'Predictions (Accuracy: {accuracy:.2%})', 
                            fontsize=12, fontweight='bold')
        
        plt.suptitle('SHAP Analysis Summary', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = 'shap_summary_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Summary analysis saved: {save_path}")
        
        # Create individual sample summaries
        self._create_sample_summaries(test_images[:5], test_labels[:5], 
                                     preds[:5], probs[:5], shap_class1[:5])
    
    def _create_sample_summaries(self, test_images, test_labels, preds, probs, shap_class1):
        """Create individual sample summaries"""
        for i in range(len(test_images)):
            try:
                true_label = test_labels[i].item()
                pred_label = preds[i].item()
                pred_prob = probs[i][1].item()
                
                # Get top features for this sample
                sample_shap = shap_class1[i]
                top_indices = np.argsort(-np.abs(sample_shap))[:15]
                top_shap_values = sample_shap[top_indices]
                
                # Create simple summary
                fig, ax = plt.subplots(figsize=(10, 6))
                
                colors = ['#2E8B57' if v > 0 else '#DC143C' for v in top_shap_values]
                bars = ax.bar(range(len(top_shap_values)), top_shap_values, 
                            color=colors, alpha=0.8, edgecolor='black')
                
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax.set_xlabel('Top Features (Ranked)', fontsize=10)
                ax.set_ylabel('SHAP Value', fontsize=10)
                ax.set_title(f'Sample {i} - Top Feature Contributions\n'
                           f'True: {"GOOD" if true_label == 1 else "BAD"} | '
                           f'Pred: {"GOOD" if pred_label == 1 else "BAD"} | '
                           f'Prob: {pred_prob:.3f}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                save_path = f'shap_force_plots/sample_{i}_summary.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"      Error creating summary for sample {i}: {e}")

def main():
    """Main execution function"""
    print("🎯 SHAP FORCE PLOT ANALYSIS")
    print("="*50)
    print(f"Creating force plots like your example image...")
    print("="*50)
    
    # Configuration
    MODEL_PATH = "best_hybrid_model1.pth"
    DATA_DIR = "/Users/tanaypatel/Documents/projects/Machine_Learning/KFU/food_classification_project/Master Dataset"
    
    try:
        # Initialize analyzer
        print("\n📦 Initializing Analyzer...")
        analyzer = SHAPForcePlotAnalyzer(
            model_path=MODEL_PATH,
            data_dir=DATA_DIR,
            num_samples=20  # Balanced samples
        )
        
        # Compute SHAP values
        print("\n⚡ Computing SHAP Values...")
        test_images, test_labels = analyzer.compute_shap_values()
        
        # Create proper force plots
        print("\n🎨 Creating Force Plots (like your example)...")
        analyzer.create_proper_force_plots(test_images, test_labels, num_samples=5)
        
        # Create summary visualizations
        print("\n📊 Creating Summary Visualizations...")
        analyzer.create_summary_visualizations(test_images, test_labels)
        
        print("\n" + "="*50)
        print("✅ ANALYSIS COMPLETE!")
        print("="*50)
        
        print(f"\n📁 YOUR FORCE PLOTS ARE READY:")
        print(f"\n1. 🎨 FORCE PLOTS (Main Output):")
        print(f"   • shap_force_plots/sample_X_force.png - Detailed force plots")
        print(f"   • shap_force_plots/sample_X_force_simple.png - Simplified versions")
        print(f"   • shap_force_plots/sample_X_summary.png - Individual summaries")
        
        print(f"\n2. 📊 SUMMARY VISUALIZATIONS:")
        print(f"   • shap_summary_analysis.png - Overall analysis")
        
        print(f"\n💡 HOW TO INTERPRET:")
        print(f"   • base value = model's expected output without features")
        print(f"   • output value = final prediction after feature contributions")
        print(f"   • Green bars = push toward GOOD classification")
        print(f"   • Red bars = push toward BAD classification")
        print(f"   • Longer bars = more influential features")
        
        print(f"\n🚀 Open the PNG files in VS Code or any image viewer!")
        
        # Cleanup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
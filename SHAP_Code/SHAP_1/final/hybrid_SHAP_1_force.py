# hybrid_SHAP_complete.py
# Complete SHAP Explainability for Hybrid Model 1

import torch
import torch.nn as nn
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
import os
import warnings
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import cv2
from tqdm import tqdm
import glob
import sys
import time
import xgboost as xgb

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
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ GPU detected!")
else:
    device = torch.device('cpu')
    print("⚠️ CPU detected.")

print(f" Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# 1. LOAD SAVED DATA
# ============================================

def load_saved_data(data_dir):
    """Load saved features and labels"""
    print("📁 Loading saved data...")
    
    X_train = np.load(os.path.join(data_dir, 'X_train_features.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_labels.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val_features.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val_labels.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_features.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_labels.npy'))
    
    print(f"✅ Data loaded successfully!")
    print(f"   Train: {X_train.shape}, {y_train.shape}")
    print(f"   Val: {X_val.shape}, {y_val.shape}")
    print(f"   Test: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ============================================
# 2. SHAP VISUALIZATION FUNCTIONS - FIXED
# ============================================

def plot_shap_summary(shap_values, X, feature_names, save_prefix="shap"):
    """Create SHAP summary plots"""
    print("📊 Generating SHAP summary plots...")
    
    try:
        # Plot 1: Beeswarm plot
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            shap_values, 
            X,
            feature_names=feature_names,
            show=False,
            max_display=20
        )
        plt.title(f"SHAP Beeswarm Plot - Feature Importance", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_beeswarm.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Plot 2: Bar plot of mean absolute SHAP
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            shap_values, 
            X,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=20
        )
        plt.title(f"SHAP Mean Absolute Values - Top 20 Features", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_mean_abs_bar.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print("✅ Summary plots created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating summary plots: {e}")

def plot_shap_heatmap(shap_values, X, feature_names, save_path="shap_heatmap.png"):
    """Create SHAP heatmap visualization"""
    print("📊 Generating SHAP heatmap...")
    
    try:
        plt.figure(figsize=(16, 12))
        
        # Create a subset for heatmap (first 50 samples for clarity)
        n_samples = min(50, shap_values.shape[0])
        shap_subset = shap_values[:n_samples]
        X_subset = X[:n_samples]
        
        # Get top 30 features by mean absolute SHAP
        mean_abs_shap = np.abs(shap_subset).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:30]
        
        shap.plots.heatmap(
            shap.Explanation(
                values=shap_subset[:, top_indices],
                data=X_subset[:, top_indices],
                feature_names=[feature_names[i] for i in top_indices]
            ),
            show=False,
            max_display=30
        )
        plt.title("SHAP Heatmap - Top 30 Features Across 50 Samples", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print("✅ Heatmap created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating heatmap: {e}")

def plot_shap_force_plots(shap_values, X, feature_names, class_names, y_true, y_pred, 
                         save_dir="force_plots"):
    """Create individual force plots for specific samples - FIXED"""
    print("📊 Generating SHAP force plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Select diverse samples
        sample_indices = []
        for i in range(2):  # For each class
            # Get correct predictions
            correct_idx = np.where((y_true == i) & (y_pred == i))[0]
            if len(correct_idx) > 0:
                sample_indices.append(correct_idx[0])
            
            # Get misclassifications
            wrong_idx = np.where((y_true == i) & (y_pred != i))[0]
            if len(wrong_idx) > 0:
                sample_indices.append(wrong_idx[0])
        
        # Add some random samples
        if len(y_true) > 4:
            random_idx = np.random.choice(len(y_true), min(4, len(y_true)), replace=False)
            sample_indices.extend(random_idx)
        
        sample_indices = list(set(sample_indices))[:8]  # Limit to 8 unique samples
        
        print(f"📊 Creating force plots for {len(sample_indices)} samples...")
        
        for idx in sample_indices:
            true_class = int(y_true[idx])
            pred_class = int(y_pred[idx])
            is_correct = true_class == pred_class
            
            # Create force plot
            plt.figure(figsize=(16, 4))
            
            shap.force_plot(
                shap.expected_value if hasattr(shap, 'expected_value') else 0,
                shap_values[idx],
                X[idx],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            status = "✓ Correct" if is_correct else "✗ Wrong"
            plt.title(f"Sample {idx}: True={class_names[true_class]}, Pred={class_names[pred_class]} ({status})",
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"force_plot_sample_{idx:04d}.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Force plots saved to {save_dir}/")
        
        # Create simplified interactive HTML
        print("📊 Creating interactive HTML visualization...")
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SHAP Force Plots - Hybrid Model Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .container { display: flex; flex-wrap: wrap; gap: 20px; }
                .sample { border: 1px solid #ddd; padding: 15px; border-radius: 8px; }
                .sample img { max-width: 100%; height: auto; }
                .correct { border-left: 5px solid #4CAF50; }
                .wrong { border-left: 5px solid #f44336; }
            </style>
        </head>
        <body>
            <h1>SHAP Force Plots Analysis</h1>
            <p>This page shows SHAP force plots for individual samples. Each plot shows how features contribute to the prediction.</p>
            <div class="container">
        """
        
        for idx in sample_indices[:6]:  # Show first 6
            true_class = int(y_true[idx])
            pred_class = int(y_pred[idx])
            is_correct = true_class == pred_class
            
            status_class = "correct" if is_correct else "wrong"
            status_text = "Correct" if is_correct else "Misclassified"
            
            html_content += f"""
            <div class="sample {status_class}">
                <h3>Sample {idx}</h3>
                <p><strong>True:</strong> {class_names[true_class]}</p>
                <p><strong>Predicted:</strong> {class_names[pred_class]}</p>
                <p><strong>Status:</strong> {status_text}</p>
                <img src="force_plot_sample_{idx:04d}.png" alt="Force plot for sample {idx}">
            </div>
            """
        
        html_content += """
            </div>
            <p><strong>How to interpret:</strong></p>
            <ul>
                <li>Red features push the prediction toward "Good"</li>
                <li>Blue features push the prediction toward "Bad"</li>
                <li>Longer arrows = stronger influence</li>
                <li>Base value = average prediction</li>
            </ul>
        </body>
        </html>
        """
        
        with open(os.path.join(save_dir, "force_plots_summary.html"), "w") as f:
            f.write(html_content)
        
        print(f"✅ Interactive HTML saved to {save_dir}/force_plots_summary.html")
        
    except Exception as e:
        print(f"❌ Error creating force plots: {e}")

def plot_shap_dependence(shap_values, X, feature_names, class_names, 
                        top_features=5, save_dir="dependence_plots"):
    """Create SHAP dependence plots for top features"""
    print("📊 Generating SHAP dependence plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Get top features by mean absolute SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_features]
        
        print(f"📊 Creating dependence plots for top {len(top_indices)} features...")
        
        for i, feat_idx in enumerate(top_indices):
            feature_name = feature_names[feat_idx]
            
            plt.figure(figsize=(12, 8))
            
            # Find another important feature for interaction
            other_indices = [idx for idx in top_indices if idx != feat_idx]
            interaction_idx = other_indices[0] if other_indices else feat_idx
            
            shap.dependence_plot(
                feat_idx,
                shap_values,
                X,
                feature_names=feature_names,
                interaction_index=interaction_idx,
                show=False,
                alpha=0.7
            )
            
            plt.title(f"Dependence Plot: Feature {feature_name}\nInteraction with {feature_names[interaction_idx]}",
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"dependence_{feature_name}.png"),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            if (i + 1) % 2 == 0:
                print(f"   Created {i + 1}/{len(top_indices)} dependence plots")
        
        print(f"✅ Dependence plots saved to {save_dir}/")
        
    except Exception as e:
        print(f"❌ Error creating dependence plots: {e}")

def plot_shap_decision(shap_values, X, feature_names, class_names, 
                      num_samples=10, save_path="shap_decision_plot.png"):
    """Create SHAP decision plot"""
    print("📊 Generating SHAP decision plot...")
    
    try:
        plt.figure(figsize=(14, 8))
        
        # Use first num_samples
        n_samples = min(num_samples, shap_values.shape[0])
        
        shap.decision_plot(
            0,  # Base value
            shap_values[:n_samples],
            X[:n_samples],
            feature_names=feature_names,
            show=False,
            highlight=0,
            feature_display_range=slice(-20, None)  # Show top 20 features
        )
        
        plt.title(f"SHAP Decision Plot - First {n_samples} Samples\n(Top 20 features shown)", 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print("✅ Decision plot created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating decision plot: {e}")

def plot_shap_waterfall(shap_values, X, feature_names, class_names, 
                       save_dir="waterfall_plots"):
    """Create SHAP waterfall plots"""
    print("📊 Generating SHAP waterfall plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Create waterfall plots for first 5 samples
        sample_indices = list(range(min(5, shap_values.shape[0])))
        
        for idx in sample_indices:
            plt.figure(figsize=(14, 8))
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[idx],
                    base_values=0,
                    data=X[idx],
                    feature_names=feature_names
                ),
                max_display=15,
                show=False
            )
            
            plt.title(f"Waterfall Plot - Sample {idx}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"waterfall_sample_{idx:04d}.png"),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Waterfall plots saved to {save_dir}/")
        
    except Exception as e:
        print(f"❌ Error creating waterfall plots: {e}")

def plot_feature_importance_comparison(xgb_model, shap_values, feature_names, 
                                      class_names, save_path="feature_importance_comparison.png"):
    """Compare XGBoost feature importance with SHAP importance"""
    print("\n📊 Generating feature importance comparison...")
    
    try:
        # Get XGBoost feature importance
        xgb_importance = xgb_model.feature_importances_
        
        # Get SHAP importance
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        # Get top 20 indices for each
        top_xgb_indices = np.argsort(xgb_importance)[::-1][:20]
        top_shap_indices = np.argsort(shap_importance)[::-1][:20]
        
        # Find common top features
        common_indices = set(top_xgb_indices[:10]) & set(top_shap_indices[:10])
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        # Plot 1: XGBoost feature importance
        axes[0, 0].barh(range(20), xgb_importance[top_xgb_indices][::-1], color='skyblue')
        axes[0, 0].set_yticks(range(20))
        axes[0, 0].set_yticklabels([feature_names[idx] for idx in top_xgb_indices[::-1]], fontsize=9)
        axes[0, 0].set_xlabel('XGBoost Importance (Gain)', fontsize=12)
        axes[0, 0].set_title('Top 20 Features - XGBoost Importance', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Plot 2: SHAP feature importance
        axes[0, 1].barh(range(20), shap_importance[top_shap_indices][::-1], color='lightcoral')
        axes[0, 1].set_yticks(range(20))
        axes[0, 1].set_yticklabels([feature_names[idx] for idx in top_shap_indices[::-1]], fontsize=9)
        axes[0, 1].set_xlabel('SHAP Mean Absolute Value', fontsize=12)
        axes[0, 1].set_title('Top 20 Features - SHAP Importance', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Correlation scatter plot
        axes[1, 0].scatter(xgb_importance, shap_importance, alpha=0.5, s=20)
        
        # Highlight top 10 features
        for idx in top_xgb_indices[:10]:
            axes[1, 0].scatter(xgb_importance[idx], shap_importance[idx], 
                             color='red', s=80, alpha=0.8, label='Top XGBoost' if idx == top_xgb_indices[0] else "")
        
        for idx in top_shap_indices[:10]:
            axes[1, 0].scatter(xgb_importance[idx], shap_importance[idx], 
                             color='blue', s=80, alpha=0.8, marker='s', 
                             label='Top SHAP' if idx == top_shap_indices[0] else "")
        
        axes[1, 0].set_xlabel('XGBoost Importance', fontsize=12)
        axes[1, 0].set_ylabel('SHAP Importance', fontsize=12)
        axes[1, 0].set_title('Correlation: XGBoost vs SHAP Feature Importance', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative importance
        sorted_xgb = np.sort(xgb_importance)[::-1]
        sorted_shap = np.sort(shap_importance)[::-1]
        
        cumulative_xgb = np.cumsum(sorted_xgb) / sorted_xgb.sum()
        cumulative_shap = np.cumsum(sorted_shap) / sorted_shap.sum()
        
        axes[1, 1].plot(range(1, len(cumulative_xgb) + 1), cumulative_xgb, 'b-', label='XGBoost', linewidth=2)
        axes[1, 1].plot(range(1, len(cumulative_shap) + 1), cumulative_shap, 'r-', label='SHAP', linewidth=2)
        axes[1, 1].axhline(y=0.8, color='gray', linestyle='--', alpha=0.7)
        axes[1, 1].axhline(y=0.9, color='gray', linestyle='--', alpha=0.7)
        
        # Find where 80% and 90% importance is reached
        xgb_80 = np.where(cumulative_xgb >= 0.8)[0][0] + 1
        shap_80 = np.where(cumulative_shap >= 0.8)[0][0] + 1
        xgb_90 = np.where(cumulative_xgb >= 0.9)[0][0] + 1
        shap_90 = np.where(cumulative_shap >= 0.9)[0][0] + 1
        
        axes[1, 1].scatter([xgb_80, shap_80], [0.8, 0.8], color=['blue', 'red'], s=100, zorder=5)
        axes[1, 1].text(xgb_80, 0.82, f' {xgb_80} features', color='blue', fontsize=10)
        axes[1, 1].text(shap_80, 0.77, f' {shap_80} features', color='red', fontsize=10)
        
        axes[1, 1].set_xlabel('Number of Features', fontsize=12)
        axes[1, 1].set_ylabel('Cumulative Importance', fontsize=12)
        axes[1, 1].set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, 500)
        
        # Calculate correlation
        correlation = np.corrcoef(xgb_importance, shap_importance)[0, 1]
        
        plt.suptitle(f'Feature Importance Comparison: XGBoost vs SHAP (Correlation: {correlation:.3f})', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"✅ Feature importance comparison created!")
        print(f"   Correlation: {correlation:.3f}")
        print(f"   Common top 10 features: {len(common_indices)}")
        
        return correlation, common_indices
        
    except Exception as e:
        print(f"❌ Error creating feature importance comparison: {e}")
        return 0, set()

def calculate_shap_metrics(shap_values, feature_names, class_names):
    """Calculate comprehensive SHAP metrics"""
    print("\n📈 Calculating SHAP metrics...")
    
    try:
        metrics = {}
        
        # Calculate basic statistics
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        mean_shap = shap_values.mean(axis=0)
        std_shap = shap_values.std(axis=0)
        
        # Get top features
        top_abs_indices = np.argsort(mean_abs_shap)[::-1][:20]
        top_positive_indices = np.argsort(mean_shap)[::-1][:10]
        top_negative_indices = np.argsort(mean_shap)[:10]
        
        # Store metrics
        metrics['top_features_abs'] = [
            {"feature": feature_names[idx], "mean_abs_shap": float(mean_abs_shap[idx])}
            for idx in top_abs_indices
        ]
        
        metrics['top_features_positive'] = [
            {"feature": feature_names[idx], "mean_shap": float(mean_shap[idx])}
            for idx in top_positive_indices
        ]
        
        metrics['top_features_negative'] = [
            {"feature": feature_names[idx], "mean_shap": float(mean_shap[idx])}
            for idx in top_negative_indices
        ]
        
        # Calculate concentration metrics
        sorted_abs = np.sort(mean_abs_shap)[::-1]
        cumulative = np.cumsum(sorted_abs) / sorted_abs.sum()
        
        metrics['concentration'] = {
            'features_50': int(np.where(cumulative >= 0.5)[0][0] + 1),
            'features_80': int(np.where(cumulative >= 0.8)[0][0] + 1),
            'features_90': int(np.where(cumulative >= 0.9)[0][0] + 1),
            'features_95': int(np.where(cumulative >= 0.95)[0][0] + 1)
        }
        
        # Print summary
        print("\n🔝 TOP 10 FEATURES BY MEAN ABSOLUTE SHAP:")
        print("-"*50)
        for i, idx in enumerate(top_abs_indices[:10], 1):
            print(f"{i:2d}. {feature_names[idx]:20s} | "
                  f"Mean |SHAP|: {mean_abs_shap[idx]:.6f} | "
                  f"Mean SHAP: {mean_shap[idx]:+.6f}")
        
        print(f"\n📊 IMPORTANCE CONCENTRATION:")
        print("-"*50)
        print(f"Features explaining 80% importance: {metrics['concentration']['features_80']}")
        print(f"Features explaining 90% importance: {metrics['concentration']['features_90']}")
        print(f"Features explaining 95% importance: {metrics['concentration']['features_95']}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error calculating SHAP metrics: {e}")
        return {}

# ============================================
# 3. MAIN EXECUTION FUNCTION
# ============================================

def main():
    print("="*70)
    print("🎯 SHAP EXPLAINABILITY FOR HYBRID MODEL 1 - WORKING VERSION")
    print("="*70)
    
    # Configuration
    DATA_DIR = "/Users/tanaypatel/Documents/projects/Machine_Learning/KFU/food_classification_project/results/hybrid_SHAP_1"
    
    # Class names
    CLASS_NAMES = ["Bad (0)", "Good (1)"]
    
    # Create output directory
    output_dir = "shap_analysis_complete"
    os.makedirs(output_dir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(output_dir)
    print(f"📁 Output will be saved to: {os.getcwd()}")
    
    # Start timing
    start_time = time.time()
    
    try:
        # 1. Load saved features
        X_train, y_train, X_val, y_val, X_test, y_test = load_saved_data(DATA_DIR)
        
        # 2. Generate feature names
        feature_dim = X_train.shape[1]
        feature_names = [f"F{i:04d}" for i in range(feature_dim)]
        print(f"📊 Feature dimension: {feature_dim}")
        
        # 3. Train surrogate XGBoost model
        print("\n🤖 Training surrogate XGBoost model...")
        
        # Combine train and val
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        
        # Train model
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        xgb_model.fit(X_combined, y_combined)
        
        # Evaluate
        y_pred_test = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        print(f"✅ Surrogate model trained!")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   Test F1-Score: {f1:.4f}")
        
        # 4. Compute SHAP values
        print("\n🧮 Computing SHAP values...")
        
        # Create explainer
        explainer = shap.TreeExplainer(xgb_model)
        
        # Use subset for efficiency
        sample_size = min(200, len(X_test))
        test_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_test_sample = X_test[test_indices]
        y_test_sample = y_test[test_indices]
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_test_sample)
        
        # Handle return format
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For class 1 (Good)
        
        print(f"✅ SHAP values computed for {sample_size} samples")
        print(f"   Shape: {shap_values.shape}")
        
        # Get predictions
        y_pred_sample = xgb_model.predict(X_test_sample)
        
        # 5. Generate all visualizations
        print("\n" + "="*70)
        print("🎨 GENERATING VISUALIZATIONS")
        print("="*70)
        
        # 5.1 Summary plots
        print("\n1. 📊 Creating summary plots...")
        plot_shap_summary(shap_values, X_test_sample, feature_names)
        
        # 5.2 Heatmap
        print("\n2. 📊 Creating heatmap...")
        plot_shap_heatmap(shap_values, X_test_sample, feature_names)
        
        # 5.3 Force plots
        print("\n3. 📊 Creating force plots...")
        plot_shap_force_plots(shap_values, X_test_sample, feature_names, CLASS_NAMES, 
                             y_test_sample, y_pred_sample)
        
        # 5.4 Dependence plots
        print("\n4. 📊 Creating dependence plots...")
        plot_shap_dependence(shap_values, X_test_sample, feature_names, CLASS_NAMES,
                            top_features=5)
        
        # 5.5 Decision plot
        print("\n5. 📊 Creating decision plot...")
        plot_shap_decision(shap_values, X_test_sample, feature_names, CLASS_NAMES,
                          num_samples=10)
        
        # 5.6 Waterfall plots
        print("\n6. 📊 Creating waterfall plots...")
        plot_shap_waterfall(shap_values, X_test_sample, feature_names, CLASS_NAMES)
        
        # 5.7 Feature importance comparison
        print("\n7. 📊 Creating feature importance comparison...")
        correlation, common_features = plot_feature_importance_comparison(
            xgb_model, shap_values, feature_names, CLASS_NAMES
        )
        
        # 5.8 Calculate metrics
        print("\n8. 📊 Calculating metrics...")
        shap_metrics = calculate_shap_metrics(shap_values, feature_names, CLASS_NAMES)
        
        # Add additional metrics
        shap_metrics['model_performance'] = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'samples_analyzed': sample_size
        }
        
        shap_metrics['importance_correlation'] = float(correlation)
        
        # Save metrics
        with open("shap_metrics.json", "w") as f:
            json.dump(shap_metrics, f, indent=2)
        print("✅ Metrics saved to shap_metrics.json")
        
        # Save data
        np.save("shap_values.npy", shap_values)
        np.save("X_sample.npy", X_test_sample)
        np.save("y_true_sample.npy", y_test_sample)
        np.save("y_pred_sample.npy", y_pred_sample)
        print("✅ Data saved for future analysis")
        
        # Generate report
        elapsed_time = time.time() - start_time
        
        report = f"""
        ===================================================
        SHAP ANALYSIS REPORT
        ===================================================
        
        ANALYSIS SUMMARY
        ----------------
        • Model: Hybrid CNN Features → XGBoost Classifier
        • Classes: {CLASS_NAMES[0]}, {CLASS_NAMES[1]}
        • Feature Dimension: {feature_dim}
        • Samples Analyzed: {sample_size}
        • Analysis Time: {elapsed_time:.1f} seconds
        
        MODEL PERFORMANCE
        -----------------
        • Accuracy:  {accuracy:.4f}
        • F1-Score:  {f1:.4f}
        
        KEY FINDINGS
        ------------
        • Feature Importance Correlation (XGBoost vs SHAP): {correlation:.3f}
        • Features explaining 80% importance: {shap_metrics.get('concentration', {}).get('features_80', 'N/A')}
        • Features explaining 90% importance: {shap_metrics.get('concentration', {}).get('features_90', 'N/A')}
        
        GENERATED FILES
        ---------------
        Visualizations:
        • shap_beeswarm.png - Beeswarm plot of feature importance
        • shap_mean_abs_bar.png - Bar plot of mean |SHAP|
        • shap_heatmap.png - Heatmap of feature contributions
        • shap_decision_plot.png - Decision plot for samples
        • feature_importance_comparison.png - XGBoost vs SHAP
        
        Detailed Plots:
        • force_plots/ - Individual force plots (PNG + HTML)
        • dependence_plots/ - Feature dependence plots
        • waterfall_plots/ - Waterfall plots
        
        Data Files:
        • shap_metrics.json - Comprehensive metrics
        • shap_values.npy - Raw SHAP values
        • X_sample.npy - Feature data used
        • y_*.npy - True and predicted labels
        
        ===================================================
        ANALYSIS COMPLETED SUCCESSFULLY!
        ===================================================
        """
        
        print(report)
        
        # Save report
        with open("analysis_report.txt", "w") as f:
            f.write(report)
        
        # List generated files
        print("\n📁 GENERATED FILES:")
        print("-"*50)
        
        file_counts = {}
        for root, dirs, files in os.walk("."):
            level = root.replace(".", "").count(os.sep)
            indent = "  " * level
            if root != ".":
                print(f"{indent}📂 {os.path.basename(root)}/")
            subindent = "  " * (level + 1)
            for file in files:
                if file.endswith(('.png', '.html', '.json', '.npy', '.txt')):
                    print(f"{subindent}• {file}")
                    ext = os.path.splitext(file)[1]
                    file_counts[ext] = file_counts.get(ext, 0) + 1
        
        print(f"\n📊 Total files generated: {sum(file_counts.values())}")
        for ext, count in file_counts.items():
            print(f"   {ext}: {count} files")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Return to original directory
        os.chdir(original_dir)
        print(f"\n📁 Analysis complete! Files saved to: {output_dir}/")
        print(f"⏱️ Total time: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()
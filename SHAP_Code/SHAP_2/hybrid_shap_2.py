# Hybrid Model 2: DeiT + MobileViT with XGBoost + SHAP Explainable AI
# Complete implementation with real-time visualization and M3 optimization

# Patch torch.dynamo to avoid FileNotFoundError on read-only systems
try:
    import torch._dynamo.config
    torch._dynamo.config.debug_dir_root = "/tmp/torch_compile_debug"
except:
    pass

import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import numpy as np
# Patch for SHAP 0.41.0 compatibility with NumPy 1.24+
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
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
import hpbandster.core.nameserver as hpns
import ConfigSpace as CS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec

# SHAP Libraries
import shap
import seaborn as sns
import pickle
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Fix Matplotlib Font Hang & Interactive Mode Blocking
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Set up matplotlib for non-interactive plotting (prevents plt.show() blocking)
plt.ioff()  # Turn OFF interactive mode
plt.switch_backend('Agg') # Use non-interactive backend
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot') # Fallback style

plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif' # Prevent font searching loops
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# Device Configuration for M3 MacBook Air
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple Silicon (MPS) detected!")
    # Safe cache clear (avoid unsupported MPS methods)
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    
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

def _find_latest_checkpoint_dir(results_root):
    try:
        dirs = [d for d in os.listdir(results_root) if d.startswith('hybrid_SHAP_2_run_')]
    except FileNotFoundError:
        return None
    dirs_sorted = sorted(dirs, reverse=True)
    for d in dirs_sorted:
        candidate = os.path.join(results_root, d)
        ckpt = os.path.join(candidate, 'best_hybrid_model2.pth')
        if os.path.exists(ckpt):
            return candidate
    return None
# ==================== SHAP EXPLAINABLE AI MODULE ====================

class DeiTMobileViT_SHAPExplainer:
    """
    SHAP explainer optimized for DeiT + MobileViT Hybrid Model
    with M3 MacBook Air memory constraints
    """
    
    def __init__(self, device='mps', output_dir='.'):
        # Normalize device to torch.device
        self.device = device if isinstance(device, torch.device) else torch.device(str(device))
        self.explainer = None
        self.shap_values = None
        self.feature_importance = None
        self.results = {}
        self.output_dir = output_dir
        
    def extract_hybrid_features(self, hybrid_model, dataloader, max_samples=80):
        """
        Extract features from DeiT + MobileViT hybrid model
        Optimized for M3 memory constraints
        """
        print(f"🔍 Extracting hybrid features (max {max_samples} samples)...")
        
        hybrid_model.eval()
        all_features = []
        all_labels = []
        all_images = []
        
        sample_count = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting Hybrid Features"):
                if sample_count >= max_samples:
                    break
                    
                images = images.to(self.device)
                
                # Process in smaller chunks for M3
                chunk_size = min(4, images.shape[0])
                for i in range(0, images.shape[0], chunk_size):
                    if sample_count >= max_samples:
                        break
                        
                    chunk = images[i:i+chunk_size]
                    
                    # Extract features from DeiT + MobileViT hybrid
                    features = hybrid_model(chunk, features_only=True)
                    
                    all_features.append(features.cpu().numpy())
                    all_labels.append(labels[i:i+chunk_size].numpy())
                    all_images.append(chunk.cpu().numpy())
                    
                    sample_count += chunk_size
                    
                    # Clear MPS cache
                    if self.device.type == 'mps' and sample_count % 20 == 0:
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass
        
        X_features = np.vstack(all_features)
        y_labels = np.concatenate(all_labels)
        X_images = np.vstack(all_images)
        
        print(f"✅ Extracted {X_features.shape[0]} samples with {X_features.shape[1]} features")
        return X_features, y_labels, X_images
    
    def analyze_model_explanations(self, xgb_model, X_features, model_name="XGBoost", max_samples=None):
        """
        Comprehensive SHAP analysis for XGBoost model
        """
        print(f"\n🧠 SHAP Analysis for {model_name}...")
        
        # Sample if max_samples is provided and less than available features
        if max_samples is not None and len(X_features) > max_samples:
            print(f"  Sampling {max_samples} from {len(X_features)} features for analysis...")
            indices = np.random.choice(len(X_features), max_samples, replace=False)
            X_sample = X_features[indices]
        else:
            X_sample = X_features
        
        # Create SHAP explainer with fallback
        try:
            # Try 1: Standard TreeExplainer
            self.explainer = shap.TreeExplainer(xgb_model)
            # Verify explainer works by testing one sample
            _ = self.explainer.shap_values(X_sample[:1])
            print("   ✅ Standard TreeExplainer successful")
            
        except Exception as e1:
            print(f"⚠️ Standard SHAP TreeExplainer failed: {e1}")
            print("🔄 Retrying with JSON serialization workaround...")
            
            try:
                # Try 2: Save to JSON and reload as Booster (fixes encoding issues with XGBoost 2.x + SHAP 0.41)
                temp_model_path = "temp_xgb_shap.json"
                xgb_model.save_model(temp_model_path)
                booster = xgb.Booster()
                booster.load_model(temp_model_path)
                
                self.explainer = shap.TreeExplainer(booster)
                _ = self.explainer.shap_values(X_sample[:1])
                
                # Cleanup
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                    
                print("   ✅ JSON-based TreeExplainer successful")
                
            except Exception as e2:
                print(f"❌ SHAP analysis completely failed: {e2}")
                if os.path.exists("temp_xgb_shap.json"):
                    os.remove("temp_xgb_shap.json")
                return None, None
        
        # Compute SHAP values in batches
        if X_sample.shape[1] > 800:  # Large feature space
            print("  Computing SHAP in batches...")
            shap_values = []
            batch_size = 8
            
            for i in range(0, len(X_sample), batch_size):
                batch = X_sample[i:i+batch_size]
                try:
                    shap_batch = self.explainer.shap_values(batch)
                    shap_values.append(shap_batch)
                except Exception as e:
                    print(f"⚠️ Error computing SHAP batch: {e}")
                    return None, None
                
                if self.device.type == 'mps':
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
            
            self.shap_values = np.vstack(shap_values)
        else:
            try:
                self.shap_values = self.explainer.shap_values(X_sample)
            except Exception as e:
                 print(f"⚠️ Error computing SHAP values: {e}")
                 return None, None
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        return self.explainer, self.shap_values
    
    def _calculate_feature_importance(self):
        """Calculate feature importance from SHAP values"""
        if self.shap_values is not None:
            shap_importance = np.abs(self.shap_values).mean(0)
            
            # Group features by source (DeiT vs MobileViT)
            self.feature_importance = pd.DataFrame({
                'feature': [f'Feature_{i}' for i in range(len(shap_importance))],
                'shap_importance': shap_importance,
                'abs_importance': np.abs(shap_importance),
                'source': ['DeiT' if i < 384 else 'MobileViT' for i in range(len(shap_importance))]
            }).sort_values('abs_importance', ascending=False)
            
            # Calculate source contributions
            source_contributions = self.feature_importance.groupby('source')['abs_importance'].sum()
            print(f"\n📊 Source Contributions:")
            for source, contribution in source_contributions.items():
                print(f"  {source}: {contribution:.2f}")
    
    def compare_deit_mobilevit_contributions(self):
        """
        Compare feature contributions from DeiT vs MobileViT
        """
        if self.feature_importance is None:
            return None
        
        # Analyze top features by source
        top_deit = self.feature_importance[self.feature_importance['source'] == 'DeiT'].head(10)
        top_mobilevit = self.feature_importance[self.feature_importance['source'] == 'MobileViT'].head(10)
        
        comparison = {
            'deit_top_features': top_deit.to_dict('records'),
            'mobilevit_top_features': top_mobilevit.to_dict('records'),
            'deit_total_importance': top_deit['abs_importance'].sum(),
            'mobilevit_total_importance': top_mobilevit['abs_importance'].sum(),
            'deit_avg_importance': top_deit['abs_importance'].mean(),
            'mobilevit_avg_importance': top_mobilevit['abs_importance'].mean()
        }
        
        return comparison
    
    def visualize_hybrid_contributions(self):
        """
        Visualize contributions from DeiT vs MobileViT
        """
        if self.feature_importance is None:
            return
        
        comparison = self.compare_deit_mobilevit_contributions()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Source Contribution Pie Chart
        deit_contrib = comparison['deit_total_importance']
        mvit_contrib = comparison['mobilevit_total_importance']
        total = deit_contrib + mvit_contrib
        
        axes[0].pie([deit_contrib, mvit_contrib], 
                   labels=[f'DeiT ({deit_contrib/total:.1%})', 
                           f'MobileViT ({mvit_contrib/total:.1%})'],
                   autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
        axes[0].set_title('Feature Importance by Source')
        
        # 2. Top DeiT Features
        deit_top = self.feature_importance[self.feature_importance['source'] == 'DeiT'].head(8)
        axes[1].barh(range(len(deit_top)), deit_top['abs_importance'].values[::-1], color='#ff9999')
        axes[1].set_yticks(range(len(deit_top)))
        axes[1].set_yticklabels(deit_top['feature'].values[::-1])
        axes[1].set_xlabel('Mean |SHAP| Value')
        axes[1].set_title('Top DeiT Features')
        
        # 3. Top MobileViT Features
        mvit_top = self.feature_importance[self.feature_importance['source'] == 'MobileViT'].head(8)
        axes[2].barh(range(len(mvit_top)), mvit_top['abs_importance'].values[::-1], color='#66b3ff')
        axes[2].set_yticks(range(len(mvit_top)))
        axes[2].set_yticklabels(mvit_top['feature'].values[::-1])
        axes[2].set_xlabel('Mean |SHAP| Value')
        axes[2].set_title('Top MobileViT Features')
        
        plt.suptitle('DeiT vs MobileViT Feature Contributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, 'shap_deit_vs_mobilevit.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparison
    
    def visualize_global_importance(self, shap_values, X_features, save_prefix="shap"):
        """
        Generate global SHAP visualizations
        """
        print("\n📊 Generating global SHAP visualizations...")
        
        base = save_prefix if os.path.isabs(save_prefix) else os.path.join(self.output_dir, save_prefix)
        
        # Summary plot
        try:
            plt.figure(figsize=(14, 8))
            shap.summary_plot(
                shap_values,
                X_features,
                max_display=20,
                show=False
            )
            plt.title("DeiT+MobileViT Hybrid - SHAP Feature Importance", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{base}_summary.png", dpi=300, bbox_inches='tight')
            plt.close() # Close to free memory
        except Exception as e:
            print(f"⚠️ Summary plot failed: {e}")
            plt.close()
        
        # Bar plot
        try:
            plt.figure(figsize=(12, 6))
            shap.summary_plot(
                shap_values,
                X_features,
                plot_type="bar",
                max_display=20,
                show=False
            )
            plt.title("DeiT+MobileViT Hybrid - Mean |SHAP| Value", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{base}_bar.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️ Bar plot failed: {e}")
            plt.close()

        
        # Dependence plot for top feature
        if shap_values.shape[1] > 0:
            try:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    0, shap_values, X_features,
                    interaction_index=None,
                    show=False
                )
                plt.title("SHAP Dependence Plot - Most Important Feature", fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"{base}_dependence.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"⚠️ Warning: Could not generate dependence plot: {e}")
                
    def explain_hybrid_prediction(self, hybrid_model, xgb_model, image_tensor, true_label=None):
        """
        Explain individual prediction with hybrid architecture insights
        """
        print("\n🔎 Explaining hybrid model prediction...")
        
        hybrid_model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device).unsqueeze(0)
            
            # Get features from hybrid model
            features = hybrid_model(image_tensor, features_only=True)
            features_np = features.cpu().numpy()
            
            # Separate DeiT and MobileViT features
            deit_features = features_np[0, :384]  # First 384 features from DeiT
            mvit_features = features_np[0, 384:]  # Remaining from MobileViT
        
        # Get prediction
        prediction = xgb_model.predict(features_np)[0]
        probability = xgb_model.predict_proba(features_np)[0]
        
        # Create explainer with fallback
        if self.explainer is None:
             try:
                 self.explainer = shap.TreeExplainer(xgb_model)
                 _ = self.explainer.shap_values(features_np)
             except Exception:
                 try:
                     temp_model_path = "temp_xgb_shap_single.json"
                     xgb_model.save_model(temp_model_path)
                     booster = xgb.Booster()
                     booster.load_model(temp_model_path)
                     self.explainer = shap.TreeExplainer(booster)
                     if os.path.exists(temp_model_path):
                         os.remove(temp_model_path)
                 except Exception as e:
                     print(f"❌ Could not initialize SHAP explainer: {e}")
                     return None
        
        shap_values = self.explainer.shap_values(features_np)
        explainer = self.explainer
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Original Image
        img_np = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        # Fix encoding issue by decoding if necessary
        try:
             # Ensure proper encoding for matplotlib
             import locale
             locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except:
             pass

        axes[0, 0].imshow(img_np)
        pred_text = f"Prediction: {'Good' if prediction == 1 else 'Bad'}\n"
        pred_text += f"Confidence: {probability[prediction]:.3f}\n"
        if true_label is not None:
            pred_text += f"True Label: {'Good' if true_label == 1 else 'Bad'}"
        axes[0, 0].set_title(pred_text, fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Force Plot
        feature_names = [f'DeiT_F{i}' for i in range(384)] + [f'MViT_F{i}' for i in range(features_np.shape[1] - 384)]
        try:
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                features_np[0],
                feature_names=feature_names[:20],
                matplotlib=True,
                show=False
            )
            axes[0, 1].set_title('SHAP Force Plot', fontsize=12, fontweight='bold')
        except Exception as e:
            print(f"⚠️ Warning: Could not generate force plot: {e}")
            axes[0, 1].text(0.5, 0.5, "Force Plot Error", ha='center')
        
        # 3. Feature Source Comparison
        source_contributions = {
            'DeiT': np.abs(shap_values[0, :384]).sum(),
            'MobileViT': np.abs(shap_values[0, 384:]).sum()
        }
        
        axes[1, 0].bar(source_contributions.keys(), source_contributions.values(), 
                      color=['#ff9999', '#66b3ff'])
        axes[1, 0].set_ylabel('Total |SHAP| Contribution')
        axes[1, 0].set_title('Contribution by Model Source', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Top Contributing Features
        top_indices = np.argsort(np.abs(shap_values[0]))[-8:][::-1]
        top_features = [f'F{i}' for i in top_indices]
        top_values = shap_values[0, top_indices]
        
        colors = ['red' if v < 0 else 'green' for v in top_values]
        axes[1, 1].barh(range(len(top_features)), top_values, color=colors)
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels(top_features)
        axes[1, 1].set_xlabel('SHAP Value')
        axes[1, 1].set_title('Top Contributing Features', fontsize=12, fontweight='bold')
        axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.suptitle('DeiT+MobileViT Hybrid Model Prediction Explanation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, 'shap_hybrid_prediction.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'prediction': prediction,
            'probability': probability,
            'shap_values': shap_values,
            'deit_contribution': source_contributions['DeiT'],
            'mobilevit_contribution': source_contributions['MobileViT']
        }
    
    def generate_hybrid_report(self, hybrid_model_name, xgb_model_name, comparison_results=None):
        """Generate comprehensive report for hybrid model"""
        
        report = {
            'analysis_info': {
                'model': hybrid_model_name,
                'xgboost_model': xgb_model_name,
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'architecture': 'DeiT + MobileViT Hybrid'
            },
            'summary': {
                'total_features': len(self.feature_importance) if self.feature_importance is not None else 0,
                'top_features': self.feature_importance.head(10).to_dict('records') if self.feature_importance is not None else []
            }
        }
        
        # Add hybrid-specific analysis
        source_comparison = self.compare_deit_mobilevit_contributions()
        if source_comparison:
            report['hybrid_analysis'] = {
                'deit_vs_mobilevit': source_comparison,
                'dominant_source': 'DeiT' if source_comparison['deit_total_importance'] > 
                source_comparison['mobilevit_total_importance'] else 'MobileViT'
            }
        
        if comparison_results:
            report['optimization_comparison'] = comparison_results
        
        # Save report
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        report_path = os.path.join(self.output_dir, 'shap_hybrid_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        # Save feature importance
        fi_path = None
        if self.feature_importance is not None:
            fi_path = os.path.join(self.output_dir, 'shap_hybrid_feature_importance.csv')
            self.feature_importance.to_csv(fi_path, index=False)
        
        print("\n📋 Hybrid SHAP Report Generated:")
        print(f"   - {report_path}")
        if fi_path:
            print(f"   - {fi_path}")
        
        return report
    
    def save_analysis(self, filename='shap_hybrid_analysis.pkl'):
        """Save complete SHAP analysis for future use"""
        analysis_data = {
            'explainer': self.explainer,
            'shap_values': self.shap_values,
            'feature_importance': self.feature_importance,
            'results': self.results
        }
        
        out_path = filename if os.path.isabs(filename) else os.path.join(self.output_dir, filename)
        with open(out_path, 'wb') as f:
            pickle.dump(analysis_data, f)
        print(f"💾 Complete SHAP analysis saved to {out_path}")

# ==================== END OF SHAP MODULE ====================

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
        main_categories = [
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.') and not d.startswith('._')
        ]
        for main_category in main_categories:
            main_category_path = os.path.join(self.data_dir, main_category)
            # If category contains direct Good/Bad folders, treat as no subcategory
            try:
                subdirs = [
                    d for d in os.listdir(main_category_path)
                    if os.path.isdir(os.path.join(main_category_path, d)) and not d.startswith('.') and not d.startswith('._')
                ]
            except FileNotFoundError:
                subdirs = []

            has_quality_direct = any(sd.lower() in ['good', 'bad'] for sd in subdirs)
            if main_category.lower() in ['egg', 'paneer', 'dairy product'] or has_quality_direct:
                folder_structure[main_category] = ['']
            else:
                folder_structure[main_category] = subdirs
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
            # Check if file path issue due to copy suffix or hidden files
            if "[Errno 2] No such file or directory" in str(e):
                # Try to clean path or find alternative
                base_dir = os.path.dirname(img_path)
                filename = os.path.basename(img_path)
                
                # Check if it's a hidden file
                if filename.startswith('._'):
                     clean_name = filename[2:]
                     clean_path = os.path.join(base_dir, clean_name)
                     if os.path.exists(clean_path):
                         try:
                             image = Image.open(clean_path).convert('RGB')
                             return self.transform(image) if self.transform else image, label
                         except:
                             pass

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
    def __init__(self, model_name, output_dir='.'):
        self.model_name = model_name
        self.output_dir = output_dir
        self.epoch_data = []
        self.boosting_results = {}
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
        
    def add_boosting_result(self, method_name, metrics):
        if not hasattr(self, 'boosting_results'):
            self.boosting_results = {}
        self.boosting_results[method_name] = metrics
        
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
        out_path = os.path.join(self.output_dir, f'{self.model_name}_final_training_history.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"✅ Final training plot saved as: {out_path}")

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

# BOHB Optimization for XGBoost
def run_bohb_optimization(X_train, y_train, X_val, y_val, X_test, y_test, history):
    print("\n" + "="*60)
    print("🔧 BOHB OPTIMIZATION FOR XGBoost")
    print("="*60)
    
    try:
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
                    random_state=42,
                    n_jobs=1
                )
                model.fit(self.X_train, self.y_train)
                val_pred = model.predict(self.X_val)
                f1 = f1_score(self.y_val, val_pred, average='weighted')
                return {'loss': 1 - f1, 'info': {'f1_score': f1}}
        
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate', 0.01, 0.3))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_depth', 3, 10))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('subsample', 0.6, 1.0))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('colsample_bytree', 0.6, 1.0))
        
        # Start a nameserver
        NS = hpns.NameServer(run_id='bohb_xgboost', host='127.0.0.1', port=None)
        ns_host, ns_port = NS.start()

        worker = XGBoostBOHBWorker(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            nameserver=ns_host, nameserver_port=ns_port,
            run_id='bohb_xgboost'
        )
        worker.run(background=True)
        
        from hpbandster.optimizers import BOHB
        bohb = BOHB(
            configspace=config_space,
            run_id='bohb_xgboost',
            nameserver=ns_host,
            nameserver_port=ns_port,
            min_budget=50,
            max_budget=200
        )
        
        results = bohb.run(n_iterations=5)
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()
        
        id2config = results.get_id2config_mapping()
        incumbent = results.get_incumbent_id()
        best_config = id2config[incumbent]['config']
        
        # Train final model
        best_xgb = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=best_config['max_depth'],
            learning_rate=best_config['learning_rate'],
            subsample=best_config['subsample'],
            colsample_bytree=best_config['colsample_bytree'],
            random_state=42
        )
        best_xgb.fit(X_train, y_train)
        
        # Save model for SHAP
        out_dir = getattr(history, 'output_dir', '.')
        bohb_path = os.path.join(out_dir, 'bohb_xgboost_hybrid2.pkl')
        with open(bohb_path, 'wb') as f:
            pickle.dump(best_xgb, f)
        print("💾 BOHB model saved for SHAP analysis")
        
        # Evaluate
        bohb_preds = {
            'train': best_xgb.predict(X_train),
            'val': best_xgb.predict(X_val),
            'test': best_xgb.predict(X_test)
        }
        bohb_probs = {
            'train': best_xgb.predict_proba(X_train),
            'val': best_xgb.predict_proba(X_val),
            'test': best_xgb.predict_proba(X_test)
        }
        
        bohb_metrics = {
            'train': calculate_metrics(y_train, bohb_preds['train'], bohb_probs['train']),
            'val': calculate_metrics(y_val, bohb_preds['val'], bohb_probs['val']),
            'test': calculate_metrics(y_test, bohb_preds['test'], bohb_probs['test'])
        }
        
        history.add_boosting_result('XGBoost_BOHB', bohb_metrics)
        
        print("\n📊 BOHB-Optimized XGBoost Results:")
        print(f"Train - Acc: {bohb_metrics['train']['accuracy']:.4f}, F1: {bohb_metrics['train']['f1']:.4f}")
        print(f"Val   - Acc: {bohb_metrics['val']['accuracy']:.4f}, F1: {bohb_metrics['val']['f1']:.4f}")
        print(f"Test  - Acc: {bohb_metrics['test']['accuracy']:.4f}, F1: {bohb_metrics['test']['f1']:.4f}")
        
        return best_xgb
        
    except Exception as e:
        print(f"❌ BOHB optimization failed: {e}")
        return None

# Optuna Optimization for XGBoost
def run_optuna_optimization(X_train, y_train, X_val, y_val, X_test, y_test, history):
    print("\n" + "="*60)
    print("🔧 OPTUNA OPTIMIZATION FOR XGBoost")
    print("="*60)
    
    def optuna_objective(trial):
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
    
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(optuna_objective, n_trials=30)
        
        print(f"🎯 Optuna Best Params: {study.best_params}")
        print(f"🎯 Optuna Best Value: {study.best_value:.4f}")
        
        best_xgb = xgb.XGBClassifier(**study.best_params, random_state=42)
        best_xgb.fit(X_train, y_train)
        
        # Save model for SHAP
        out_dir = getattr(history, 'output_dir', '.')
        optuna_path = os.path.join(out_dir, 'optuna_xgboost_hybrid2.pkl')
        with open(optuna_path, 'wb') as f:
            pickle.dump(best_xgb, f)
        print("💾 Optuna model saved for SHAP analysis")
        
        # Evaluate
        optuna_preds = {
            'train': best_xgb.predict(X_train),
            'val': best_xgb.predict(X_val),
            'test': best_xgb.predict(X_test)
        }
        optuna_probs = {
            'train': best_xgb.predict_proba(X_train),
            'val': best_xgb.predict_proba(X_val),
            'test': best_xgb.predict_proba(X_test)
        }
        
        optuna_metrics = {
            'train': calculate_metrics(y_train, optuna_preds['train'], optuna_probs['train']),
            'val': calculate_metrics(y_val, optuna_preds['val'], optuna_probs['val']),
            'test': calculate_metrics(y_test, optuna_preds['test'], optuna_probs['test'])
        }
        
        history.add_boosting_result('XGBoost_Optuna', optuna_metrics)
        
        print("\n📊 Optuna-Optimized XGBoost Results:")
        print(f"Train - Acc: {optuna_metrics['train']['accuracy']:.4f}, F1: {optuna_metrics['train']['f1']:.4f}")
        print(f"Val   - Acc: {optuna_metrics['val']['accuracy']:.4f}, F1: {optuna_metrics['val']['f1']:.4f}")
        print(f"Test  - Acc: {optuna_metrics['test']['accuracy']:.4f}, F1: {optuna_metrics['test']['f1']:.4f}")
        
        return best_xgb
        
    except Exception as e:
        print(f"❌ Optuna optimization failed: {e}")
        return None

# ==================== SHAP INTEGRATED EXECUTION ====================

def run_shap_hybrid_analysis(hybrid_model, xgb_models, dataloaders, X_features_dict, output_dir='.', shap_samples=50):
    """
    Comprehensive SHAP analysis for DeiT + MobileViT hybrid model
    """
    print("\n" + "="*80)
    print("🧠 STEP 8: SHAP EXPLAINABLE AI ANALYSIS (DeiT+MobileViT)")
    print("="*80)
    
    # Initialize SHAP explainer for hybrid model
    shap_explainer = DeiTMobileViT_SHAPExplainer(device=device, output_dir=output_dir)
    
    # 1. Extract hybrid features for SHAP
    print(f"\n🔍 1. Extracting hybrid features for SHAP analysis ({shap_samples} samples)...")
    X_shap, y_shap, images_shap = shap_explainer.extract_hybrid_features(
        hybrid_model, dataloaders['test'], max_samples=shap_samples
    )
    
    # 2. Analyze default XGBoost model
    print("\n🌲 2. Analyzing default XGBoost with SHAP...")
    if 'default' in xgb_models:
        explainer, shap_values = shap_explainer.analyze_model_explanations(
            xgb_models['default'], X_shap, "Default_XGBoost"
        )
        
        if shap_values is not None:
            # Visualize global importance
            shap_explainer.visualize_global_importance(
                shap_values, X_shap, save_prefix=os.path.join(output_dir, "shap_hybrid_global")
            )
            
            # Analyze DeiT vs MobileViT contributions
            print("\n⚖️ 3. Analyzing DeiT vs MobileViT contributions...")
            source_comparison = shap_explainer.visualize_hybrid_contributions()
            
            if source_comparison:
                dominant = "DeiT" if source_comparison['deit_total_importance'] > source_comparison['mobilevit_total_importance'] else "MobileViT"
                print(f"  Dominant feature source: {dominant}")
                print(f"  DeiT contribution ratio: {source_comparison['deit_total_importance']/(source_comparison['deit_total_importance'] + source_comparison['mobilevit_total_importance']):.2%}")
        else:
             print("⚠️ Skipping SHAP visualizations due to previous errors.")
    
    # 3. Compare optimized models
    print("\n⚖️ 4. Comparing BOHB vs Optuna models...")
    comparison_results = None
    if 'bohb' in xgb_models and 'optuna' in xgb_models:
        # Create comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Get predictions for both models
        bohb_pred = xgb_models['bohb'].predict(X_shap[:20])
        optuna_pred = xgb_models['optuna'].predict(X_shap[:20])
        
        # Agreement analysis
        agreement = np.mean(bohb_pred == optuna_pred)
        axes[0].pie([agreement, 1-agreement], 
                   labels=[f'Agree ({agreement:.1%})', f'Disagree ({1-agreement:.1%})'],
                   autopct='%1.1f%%', colors=['green', 'red'])
        axes[0].set_title('Model Prediction Agreement')
        
        # Confidence comparison
        bohb_proba = xgb_models['bohb'].predict_proba(X_shap[:20])[:, 1]
        optuna_proba = xgb_models['optuna'].predict_proba(X_shap[:20])[:, 1]
        
        axes[1].scatter(bohb_proba, optuna_proba, alpha=0.6)
        axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1].set_xlabel('BOHB Model Confidence')
        axes[1].set_ylabel('Optuna Model Confidence')
        axes[1].set_title('Prediction Confidence Comparison')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('BOHB vs Optuna Model Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_hybrid_model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        comparison_results = {
            'agreement': agreement,
            'bohb_confidence_mean': np.mean(bohb_proba),
            'optuna_confidence_mean': np.mean(optuna_proba)
        }
    
    # 4. Individual prediction explanations
    print("\n🔎 5. Generating individual prediction explanations...")
    if 'default' in xgb_models and len(images_shap) > 0:
        # Explain first 2 samples
        for i in range(min(2, len(images_shap))):
            image_tensor = torch.tensor(images_shap[i])
            true_label = y_shap[i] if i < len(y_shap) else None
            
            result = shap_explainer.explain_hybrid_prediction(
                hybrid_model, xgb_models['default'],
                image_tensor, true_label=true_label
            )
            
            print(f"  Sample {i}: Prediction={'Good' if result['prediction']==1 else 'Bad'}, "
                  f"DeiT contrib={result['deit_contribution']:.3f}, "
                  f"MobileViT contrib={result['mobilevit_contribution']:.3f}")
    
    # 5. Generate comprehensive hybrid report
    print("\n📋 6. Generating comprehensive hybrid SHAP report...")
    report = shap_explainer.generate_hybrid_report(
        hybrid_model_name="DeiT_MobileViT_Hybrid",
        xgb_model_name="XGBoost_Hybrid_Optimized",
        comparison_results=comparison_results
    )
    
    # 6. Save complete analysis
    shap_explainer.save_analysis('shap_hybrid_complete.pkl')
    
    print("\n✅ Hybrid SHAP Analysis Completed!")
    print("📊 Generated visualizations:")
    print("   - shap_hybrid_global_summary.png")
    print("   - shap_hybrid_global_bar.png")
    print("   - shap_hybrid_global_dependence.png")
    print("   - shap_deit_vs_mobilevit.png")
    print("   - shap_hybrid_prediction.png")
    print("   - shap_hybrid_model_comparison.png")
    
    return shap_explainer

# Main Execution
def main():
    print("🚀 HYBRID MODEL 2: DeiT + MobileViT with SHAP Explainable AI")
    print("="*70)
    
    # M3 Optimized Configuration
    M3_CONFIG = {
        'batch_size': 8,  # Smaller for M3 memory
        'num_workers': 2,
        'pin_memory': False if device.type == 'mps' else True,
        'epochs': 8,  # Run full training as requested
        'image_size': 224,
        'shap_samples': 50,
    }

    # Prepare results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_root = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_root, exist_ok=True)
    output_dir = os.path.join(results_root, f'hybrid_SHAP_2_run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Cleanup conflicting files from previous runs in repo root
    repo_root = os.path.dirname(__file__)
    conflicting_files = [
        'best_hybrid_model2.pth',
        'default_xgboost_hybrid2.pkl',
        'final_hybrid2_models.pkl',
        'hybrid2_features_train.npy',
        'hybrid2_labels_train.npy',
        'hybrid2_features_val.npy',
        'hybrid2_labels_val.npy',
        'hybrid2_features_test.npy',
        'hybrid2_labels_test.npy',
        'DeiT_MobileViT_Hybrid_final_training_history.png',
        'shap_deit_vs_mobilevit.png',
        'shap_summary.png',
        'shap_bar.png',
        'shap_dependence.png',
        'shap_hybrid_prediction.png',
        'shap_hybrid_report.json',
        'shap_hybrid_feature_importance.csv',
        'shap_hybrid_analysis.pkl',
    ]
    removed = []
    for fname in conflicting_files:
        fpath = os.path.join(repo_root, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                removed.append(fname)
            except Exception as e:
                print(f"⚠️ Could not remove {fname}: {e}")
    if removed:
        print(f"🧹 Removed conflicting files: {', '.join(removed)}")
    
    # Load dataset
    print("\n📁 STEP 1: LOADING DATASET...")
    data_dir = "/Volumes/T7/KFU/Nov14/KFU_Dataset/Baked Goods"
    dataset = FoodDataset(data_dir, transform=train_transform, is_training=True)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Data loaders with M3 optimization
    batch_size = M3_CONFIG['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model and history
    print("\n🤖 STEP 2: INITIALIZING DEIT + MOBILEVIT HYBRID MODEL...")
    model = HybridModel2(num_classes=2).to(device)
    history = TrainingHistory("DeiT_MobileViT_Hybrid", output_dir=output_dir)
    
    # Training configuration
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    total_epochs = M3_CONFIG['epochs']
    deit_epochs = 4
    best_val_f1 = 0
    
    # Main Training Loop with Live Visualization
    print("\n🎯 STEP 3: CHECKING FOR EXISTING CHECKPOINTS...")
    print("="*70)

    existing_results_dir = _find_latest_checkpoint_dir(results_root)
    existing_model_path = os.path.join(existing_results_dir, "best_hybrid_model2.pth") if existing_results_dir else ""
    
    xgb_models = {}
    
    if existing_results_dir and os.path.exists(existing_model_path):
        print(f"✅ Found existing model checkpoint: {existing_model_path}")
        print("⏭️ Skipping training and loading pre-trained model...")
        model.load_state_dict(torch.load(existing_model_path, map_location=device))
        
        # Copy existing files to new output directory
        import shutil
        try:
            for file in os.listdir(existing_results_dir):
                if file.endswith(('.pth', '.pkl', '.npy', '.png', '.json', '.csv')):
                    src = os.path.join(existing_results_dir, file)
                    dst = os.path.join(output_dir, file)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
            print(f"✅ Copied existing result files to {output_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Could not copy some existing files: {e}")
            
        try:
            default_pkl = os.path.join(output_dir, 'default_xgboost_hybrid2.pkl')
            bohb_pkl = os.path.join(output_dir, 'bohb_xgboost_hybrid2.pkl')
            optuna_pkl = os.path.join(output_dir, 'optuna_xgboost_hybrid2.pkl')
            if os.path.exists(default_pkl):
                with open(default_pkl, 'rb') as f:
                    xgb_models['default'] = pickle.load(f)
            if os.path.exists(bohb_pkl):
                with open(bohb_pkl, 'rb') as f:
                    xgb_models['bohb'] = pickle.load(f)
            if os.path.exists(optuna_pkl):
                with open(optuna_pkl, 'rb') as f:
                    xgb_models['optuna'] = pickle.load(f)
            if xgb_models:
                print(f"✅ Loaded XGBoost models from {output_dir}: {', '.join(xgb_models.keys())}")
        except Exception as e:
            print(f"⚠️ Could not load XGBoost models from checkpoint: {e}")
            
    else:
        print("❌ No existing checkpoint found. Starting fresh training...")
        for epoch in range(total_epochs):
            epoch_start = time.time()
            
            # Clear MPS cache
            if device.type == 'mps':
                torch.mps.empty_cache()
            
            # Phase-based training
            if epoch < deit_epochs:
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
                best_model_path = os.path.join(output_dir, 'best_hybrid_model2.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ New best model saved: {best_model_path} (Validation F1: {best_val_f1:.4f})")
    
    print("\n" + "="*80)
    print("🎯 DEEP LEARNING PHASE COMPLETED!")
    print("="*80)
    
    # Save final plot
    history.save_final_plot()
    
    # Turn off interactive mode (redundant with initial setup but safe)
    plt.ioff()
    plt.close('all')
    
    # Feature extraction
    print("\n🔍 STEP 4: EXTRACTING HYBRID FEATURES...")
    
    # Check if features already exist in output_dir (copied from existing run)
    features_exist = all(os.path.exists(os.path.join(output_dir, f)) for f in [
        'hybrid2_features_train.npy', 'hybrid2_labels_train.npy',
        'hybrid2_features_val.npy', 'hybrid2_labels_val.npy',
        'hybrid2_features_test.npy', 'hybrid2_labels_test.npy'
    ])
    
    if features_exist:
        print("✅ Found existing features, loading from disk...")
        X_train = np.load(os.path.join(output_dir, 'hybrid2_features_train.npy'))
        y_train = np.load(os.path.join(output_dir, 'hybrid2_labels_train.npy'))
        X_val = np.load(os.path.join(output_dir, 'hybrid2_features_val.npy'))
        y_val = np.load(os.path.join(output_dir, 'hybrid2_labels_val.npy'))
        X_test = np.load(os.path.join(output_dir, 'hybrid2_features_test.npy'))
        y_test = np.load(os.path.join(output_dir, 'hybrid2_labels_test.npy'))
    else:
        best_model_path = os.path.join(output_dir, 'best_hybrid_model2.pth')
        model.load_state_dict(torch.load(best_model_path))
        X_train, y_train = extract_features(model, train_loader, device)
        X_val, y_val = extract_features(model, val_loader, device)
        X_test, y_test = extract_features(model, test_loader, device)
        
        # Save features
        np.save(os.path.join(output_dir, 'hybrid2_features_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'hybrid2_labels_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'hybrid2_features_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'hybrid2_labels_val.npy'), y_val)
        np.save(os.path.join(output_dir, 'hybrid2_features_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'hybrid2_labels_test.npy'), y_test)
        print(f"✅ Features saved to: {output_dir}")
    
    X_features_dict = {
        'train': X_train, 'y_train': y_train,
        'val': X_val, 'y_val': y_val,
        'test': X_test, 'y_test': y_test
    }
    
    print(f"Feature shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # XGBoost Training
    if 'default' not in xgb_models:
        print("\n🌲 STEP 5: TRAINING XGBOOST...")
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_model.fit(X_train, y_train)
        
        # Save default model
        with open(os.path.join(output_dir, 'default_xgboost_hybrid2.pkl'), 'wb') as f:
            pickle.dump(xgb_model, f)
        
        xgb_models['default'] = xgb_model
    else:
        print("\n🌲 STEP 5: XGBoost training skipped (loaded from checkpoint)")
        xgb_model = xgb_models['default']
    
    # Evaluate XGBoost
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_train_proba = xgb_model.predict_proba(X_train)
    xgb_test_pred = xgb_model.predict(X_test)
    xgb_test_proba = xgb_model.predict_proba(X_test)
    
    xgb_metrics = {
        'train': calculate_metrics(y_train, xgb_train_pred, xgb_train_proba),
        'test': calculate_metrics(y_test, xgb_test_pred, xgb_test_proba)
    }
    
    print("\n📊 Default XGBoost Results:")
    print(f"Train - Acc: {xgb_metrics['train']['accuracy']:.4f}, F1: {xgb_metrics['train']['f1']:.4f}")
    print(f"Test  - Acc: {xgb_metrics['test']['accuracy']:.4f}, F1: {xgb_metrics['test']['f1']:.4f}")
    
    # Optimization
    print("\n🔧 STEP 6: RUNNING OPTIMIZATIONS...")
    if 'bohb' not in xgb_models:
        bohb_model = run_bohb_optimization(X_train, y_train, X_val, y_val, X_test, y_test, history)
        if bohb_model is not None:
            xgb_models['bohb'] = bohb_model
    else:
        print("   BOHB optimization skipped (loaded from checkpoint)")
        
    if 'optuna' not in xgb_models:
        optuna_model = run_optuna_optimization(X_train, y_train, X_val, y_val, X_test, y_test, history)
        if optuna_model is not None:
            xgb_models['optuna'] = optuna_model
    else:
        print("   Optuna optimization skipped (loaded from checkpoint)")
    
    # SHAP Explainable AI Analysis
    print("\n" + "="*80)
    print("🧠 STEP 7: SHAP EXPLAINABLE AI ANALYSIS")
    print("="*80)
    
    try:
        print(f"Attempting SHAP analysis with {M3_CONFIG['shap_samples']} samples...")
        shap_explainer = run_shap_hybrid_analysis(
            model, xgb_models, dataloaders, X_features_dict, 
            output_dir=output_dir, shap_samples=M3_CONFIG['shap_samples']
        )
    except Exception as e:
        print(f"⚠️ SHAP analysis with {M3_CONFIG['shap_samples']} samples failed: {e}")
        print("🔄 Retrying with 30 samples (fallback)...")
        try:
            shap_explainer = run_shap_hybrid_analysis(
                model, xgb_models, dataloaders, X_features_dict, 
                output_dir=output_dir, shap_samples=30
            )
        except Exception as e2:
            print(f"❌ SHAP analysis fallback failed: {e2}")
            shap_explainer = None
    
    # Final Summary
    print("\n" + "="*80)
    print("🎉 HYBRID MODEL 2 - COMPLETE ANALYSIS SUMMARY")
    print("="*80)
    print("Architecture: DeiT + MobileViT Hybrid")
    print(f"Device: {device}")
    print(f"Total Features: {X_train.shape[1]} (DeiT: 384 + MobileViT: 640)")
    print(f"Total Training Time: ~{sum([d['time'] for d in history.epoch_data]):.1f}s")
    
    if shap_explainer and shap_explainer.feature_importance is not None:
        # Get DeiT vs MobileViT analysis
        source_comparison = shap_explainer.compare_deit_mobilevit_contributions()
        
        if source_comparison:
            print(f"\n📊 Source Analysis:")
            print(f"  DeiT Contribution: {source_comparison['deit_total_importance']:.2f}")
            print(f"  MobileViT Contribution: {source_comparison['mobilevit_total_importance']:.2f}")
            print(f"  Dominant Source: {'DeiT' if source_comparison['deit_total_importance'] > source_comparison['mobilevit_total_importance'] else 'MobileViT'}")
            
            print(f"\n🔝 Top 3 DeiT Features:")
            for i, feat in enumerate(source_comparison['deit_top_features'][:3], 1):
                print(f"  {i}. {feat['feature']}: {feat['abs_importance']:.4f}")
            
            print(f"\n🔝 Top 3 MobileViT Features:")
            for i, feat in enumerate(source_comparison['mobilevit_top_features'][:3], 1):
                print(f"  {i}. {feat['feature']}: {feat['abs_importance']:.4f}")
    
    print("\n📊 XGBoost Performance Summary:")
    for method, results in history.boosting_results.items():
        print(f"\n{method}:")
        if 'train' in results:
            print(f"  Train - Acc: {results['train']['accuracy']:.4f}, F1: {results['train']['f1']:.4f}")
        if 'test' in results:
            print(f"  Test  - Acc: {results['test']['accuracy']:.4f}, F1: {results['test']['f1']:.4f}")
    
    print(f"\n✅ DeiT + MobileViT Hybrid Model with SHAP completed!")
    print(f"📊 Training curves: {os.path.join(output_dir, 'DeiT_MobileViT_Hybrid_final_training_history.png')}")
    print(f"🧠 SHAP analysis: {os.path.join(output_dir, 'shap_hybrid_report.json')}")
    print(f"💾 Features saved as: {os.path.join(output_dir, 'hybrid2_features_*.npy')}")
    
    # Save final models
    with open(os.path.join(output_dir, 'final_hybrid2_models.pkl'), 'wb') as f:
        pickle.dump({
            'hybrid_model': model.state_dict(),
            'xgb_models': xgb_models,
            'shap_explainer': shap_explainer
        }, f)
    
    print(f"💾 All models saved to {os.path.join(output_dir, 'final_hybrid2_models.pkl')}")
    
    # Close all figures
    plt.close('all')

if __name__ == "__main__":
    main()
"""
Saliency Heatmaps for Multiple Food Categories (Excluding Egg)
Uses input gradients – lightweight, no hooks, MPS‑friendly.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import sys
from collections import defaultdict

# Import your model and dataset class
sys.path.append('.')
from hybrid_model_1 import HybridModel1, FoodDataset

# ------------------------- Config -------------------------
MODEL_PATH = "best_hybrid_model1.pth"
DATA_DIR = "/Users/tanaypatel/Documents/projects/Machine_Learning/KFU/food_classification_project/Master Dataset"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUTPUT_DIR = "saliency_overlays_categories"

# Categories to include (all except Egg)
CATEGORIES = [
    "Baked Goods_Bread", "Baked Goods_CAKE", "Baked Goods_Dhokla",
    "Baked Goods_Laddu", "Baked Goods_Samosa",
    "Fish_Chingri", "Fish_MOYA FISH", "Fish_POMFRET", "Fish_Pona fish",
    "Grains_Good And Bad Classification Of Boiled Rice",
    "Meat Product_Good And Bad Classification Of Chicken Liver",
    "Pickles_Kuler achar", "paneer"
]

IMAGES_PER_CATEGORY = 2          # total per category (1 good + 1 bad)
SALIENCY_NORMALIZE = True        # normalize saliency map to [0,1] for display

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# ------------------------- Model Loading -------------------------
def load_model():
    model = HybridModel1(num_classes=2).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")
    return model

# ------------------------- Dataset with Category Info -------------------------
def get_dataset_with_categories():
    """Return dataset and a mapping from index -> (category, label)."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = FoodDataset(DATA_DIR, transform=transform, is_training=False)
    
    # Build category mapping from class_name (e.g., "Baked Goods_Bread_good")
    idx_info = []
    for i in range(len(dataset)):
        class_name = dataset.class_names[i]
        # class_name format: "Category[_subcategory]_quality"
        # Remove the last _good or _bad to get category
        if class_name.endswith('_good'):
            category = class_name[:-5]   # remove '_good'
            label = 1
        elif class_name.endswith('_bad'):
            category = class_name[:-4]   # remove '_bad'
            label = 0
        else:
            # fallback: use path parsing
            category = "Unknown"
            label = dataset.labels[i]
        idx_info.append((category, label))
    
    return dataset, idx_info

# ------------------------- Saliency Computation -------------------------
def compute_saliency(model, img_tensor, target_class=1):
    """
    Compute input gradients for a target class using LOGITS (not probability).
    Returns a visible saliency map and the predicted probability.
    """
    img = img_tensor.clone().detach().requires_grad_(True).unsqueeze(0).to(DEVICE)
    logits = model(img)                     # shape (1,2)
    # Use logit of the target class (before softmax) – avoids vanishing gradients
    target_logit = logits[0, target_class]
    model.zero_grad()
    grad = torch.autograd.grad(target_logit, img, retain_graph=False, create_graph=False)[0]
    grad_np = grad.detach().cpu().numpy()[0]          # (3,224,224)
    
    # Combine channels: sum of absolute gradients (more sensitive than max)
    saliency = np.sum(np.abs(grad_np), axis=0)        # (224,224)
    
    # Contrast stretch: clip top 5% to improve visualisation
    p95 = np.percentile(saliency, 95)
    saliency = np.clip(saliency, 0, p95)
    if saliency.max() > saliency.min():
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    
    # Also compute the actual probability (for display)
    probs = torch.softmax(logits, dim=1)[0]
    prob_target = probs[target_class].item()
    
    # Cleanup
    del img, grad
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return saliency, prob_target

# ------------------------- Denormalize Image -------------------------
def denormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = img_tensor.numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)

# ------------------------- Overlay Creation -------------------------
def create_overlay(original, saliency, alpha=0.6, cmap='jet'):
    """Original: (H,W,3) in [0,1]; saliency: (H,W) in [0,1]"""
    heatmap = plt.cm.get_cmap(cmap)(saliency)[:, :, :3]
    overlay = (1 - alpha) * original + alpha * heatmap
    return np.clip(overlay, 0, 1)

# ------------------------- Main -------------------------
def main():
    print("\n" + "="*70)
    print("SALIENCY HEATMAPS FOR MULTIPLE CATEGORIES (EXCLUDING EGG)")
    print("="*70)
    
    # 1. Load model
    model = load_model()
    
    # 2. Load dataset with category info
    dataset, idx_info = get_dataset_with_categories()
    print(f"Total images in dataset: {len(dataset)}")
    
    # 3. Group indices by category and label
    category_indices = defaultdict(lambda: {'good': [], 'bad': []})
    for i, (cat, label) in enumerate(idx_info):
        if cat in CATEGORIES:
            if label == 1:
                category_indices[cat]['good'].append(i)
            else:
                category_indices[cat]['bad'].append(i)
    
    # 4. Select balanced samples per category (1 good + 1 bad, or as available)
    selected = []  # list of (image_index, category, label)
    for cat in CATEGORIES:
        goods = category_indices[cat]['good'][:IMAGES_PER_CATEGORY//2]
        bads  = category_indices[cat]['bad'][:IMAGES_PER_CATEGORY - len(goods)]
        for idx in goods:
            selected.append((idx, cat, 1))
        for idx in bads:
            selected.append((idx, cat, 0))
        print(f"{cat}: {len(goods)} good, {len(bads)} bad")
    
    if not selected:
        print("No images found for the specified categories. Check category names.")
        return
    
    print(f"\nGenerating heatmaps for {len(selected)} images...")
    
    # 5. Process each selected image
    for order, (img_idx, category, true_label) in enumerate(selected):
        print(f"\n[{order+1}/{len(selected)}] Category: {category} | True: {'GOOD' if true_label==1 else 'BAD'}")
        
        img_tensor = dataset[img_idx][0]  # already transformed
        img_original = denormalize(img_tensor)
        
        # Compute saliency (target class = 1 = good)
        saliency, prob_good = compute_saliency(model, img_tensor, target_class=1)
        
        # Prediction (which class has higher probability)
        pred_label = 1 if prob_good > 0.5 else 0
        confidence = prob_good if pred_label == 1 else (1 - prob_good)
        
        print(f"   Pred: {'GOOD' if pred_label==1 else 'BAD'} (prob(good)={prob_good:.3f})")
        
        # Create overlay
        overlay = create_overlay(img_original, saliency, alpha=0.6)
        
        # Save side-by-side figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_original)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        im = axes[1].imshow(saliency, cmap='jet')
        axes[1].set_title("Saliency Map\n(importance for GOOD)")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[2].imshow(overlay)
        title = f"{category}\nTrue: {'GOOD' if true_label==1 else 'BAD'} | "
        title += f"Pred: {'GOOD' if pred_label==1 else 'BAD'} ({confidence:.1%})"
        axes[2].set_title(title, fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        safe_cat = category.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(OUTPUT_DIR, f"{safe_cat}_img{order}_saliency.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {out_path}")
        
        # Also save standalone heatmap (optional)
        plt.figure(figsize=(8,6))
        plt.imshow(saliency, cmap='jet')
        plt.colorbar()
        plt.title(f"{category} - Saliency (True: {'GOOD' if true_label==1 else 'BAD'})")
        plt.axis('off')
        heatmap_path = os.path.join(OUTPUT_DIR, f"{safe_cat}_img{order}_heatmap.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n✅ All saliency overlays saved in '{OUTPUT_DIR}/'")
    print("Red regions = pixels that push the model toward GOOD classification.")

if __name__ == "__main__":
    main()
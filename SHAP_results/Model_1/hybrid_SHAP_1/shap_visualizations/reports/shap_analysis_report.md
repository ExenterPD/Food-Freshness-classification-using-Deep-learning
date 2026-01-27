# SHAP Analysis Report - Hybrid Model 1

## Model Information
- **Model Name**: HybridModel1 (EfficientNetV2M + MobileViT)
- **Device**: mps
- **Classes**: Bad Quality, Good Quality
- **Total Samples**: 14,321

## SHAP Analysis Results
- **Total Features Analyzed**: 1920
- **Mean Absolute SHAP**: 0.004994
- **Standard Deviation**: 0.068493
- **Maximum Value**: 1.965416
- **Minimum Value**: 0.000000

## Top 10 Most Important Features
| Rank | Feature Index | Feature Name | Mean Absolute SHAP |
|------|---------------|--------------|---------------------|
| 1 | 219 | Feature_219 | 1.965416 |
| 2 | 433 | Feature_433 | 1.628870 |
| 3 | 79 | Feature_79 | 0.903643 |
| 4 | 732 | Feature_732 | 0.717930 |
| 5 | 110 | Feature_110 | 0.697665 |
| 6 | 812 | Feature_812 | 0.487148 |
| 7 | 1152 | Feature_1152 | 0.298928 |
| 8 | 790 | Feature_790 | 0.289155 |
| 9 | 1054 | Feature_1054 | 0.281277 |
| 10 | 857 | Feature_857 | 0.223643 |


## Generated Visualizations

### Summary Plots
- `shap_summary.png` - SHAP summary plot showing feature importance
- `mean_absolute_shap.png` - Bar plot of mean absolute SHAP values

### Force Plots
- `force_plot_sample_1_correct_confident.png`
- `force_plot_sample_17_correct_confident.png`
- `force_plot_sample_75_correct_confident.png`
- `force_plot_sample_11_correct_confident.png`
- `force_plot_sample_6_correct_confident.png`


### Heatmaps
- `heatmap_sample_2.png`
- `heatmap_sample_0.png`
- `heatmap_sample_1.png`


### Overlays
- `overlay_sample_1.png`
- `overlay_sample_0.png`
- `overlay_sample_2.png`


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

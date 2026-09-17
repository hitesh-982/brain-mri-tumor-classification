# Qualitative Error Analysis Report — Baseline_Attention_CNN

## Summary Statistics
- **Total Test Samples**: 1034
- **Correct Predictions**: 887 (85.78%)
- **Misclassifications**: 147 (14.22%)

## Confused Tumor Class Patterns
Below are representative misclassified samples and confidence breakdown:

### Sample 1: brisc2025_train_04887_pi_sa_t1.jpg
- **True Class**: `pituitary`
- **Predicted Class**: `notumor` (Confidence: 55.78%)
- **Probabilities**: Glioma: 0.017, Meningioma: 0.067, NoTumor: 0.558, Pituitary: 0.358

### Sample 2: Te-aug-me_75.jpg
- **True Class**: `meningioma`
- **Predicted Class**: `notumor` (Confidence: 75.90%)
- **Probabilities**: Glioma: 0.054, Meningioma: 0.140, NoTumor: 0.759, Pituitary: 0.047

### Sample 3: brisc2025_test_00151_gl_co_t1.jpg
- **True Class**: `glioma`
- **Predicted Class**: `meningioma` (Confidence: 61.05%)
- **Probabilities**: Glioma: 0.370, Meningioma: 0.611, NoTumor: 0.014, Pituitary: 0.006

### Sample 4: Te-gl_32.jpg
- **True Class**: `glioma`
- **Predicted Class**: `notumor` (Confidence: 68.68%)
- **Probabilities**: Glioma: 0.083, Meningioma: 0.188, NoTumor: 0.687, Pituitary: 0.042

### Sample 5: brisc2025_train_01041_gl_sa_t1.jpg
- **True Class**: `glioma`
- **Predicted Class**: `meningioma` (Confidence: 62.89%)
- **Probabilities**: Glioma: 0.342, Meningioma: 0.629, NoTumor: 0.003, Pituitary: 0.026

### Sample 6: brisc2025_train_00822_gl_co_t1.jpg
- **True Class**: `glioma`
- **Predicted Class**: `meningioma` (Confidence: 82.88%)
- **Probabilities**: Glioma: 0.169, Meningioma: 0.829, NoTumor: 0.001, Pituitary: 0.001

### Sample 7: brisc2025_test_00164_gl_co_t1.jpg
- **True Class**: `glioma`
- **Predicted Class**: `meningioma` (Confidence: 86.32%)
- **Probabilities**: Glioma: 0.133, Meningioma: 0.863, NoTumor: 0.001, Pituitary: 0.003

### Sample 8: brisc2025_test_00433_me_co_t1.jpg
- **True Class**: `meningioma`
- **Predicted Class**: `pituitary` (Confidence: 87.18%)
- **Probabilities**: Glioma: 0.009, Meningioma: 0.119, NoTumor: 0.001, Pituitary: 0.872

### Sample 9: brisc2025_test_00411_me_co_t1.jpg
- **True Class**: `meningioma`
- **Predicted Class**: `notumor` (Confidence: 65.08%)
- **Probabilities**: Glioma: 0.074, Meningioma: 0.258, NoTumor: 0.651, Pituitary: 0.017

### Sample 10: brisc2025_train_00331_gl_ax_t1.jpg
- **True Class**: `glioma`
- **Predicted Class**: `meningioma` (Confidence: 69.67%)
- **Probabilities**: Glioma: 0.285, Meningioma: 0.697, NoTumor: 0.008, Pituitary: 0.011


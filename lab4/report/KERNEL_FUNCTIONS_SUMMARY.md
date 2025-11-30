# SVM Kernel Functions: Impact on Classification Results

**Authors:** Kamil Suchomski, Kamil Koniak

## Executive Summary

This document summarizes how different Support Vector Machine (SVM) kernel functions and their parameters affect classification results on two datasets: Banknote Authentication and Pima Indians Diabetes.

## Kernel Functions Tested

### 1. Linear Kernel
**Formula:** K(x, y) = x^T · y

**Parameters:**
- `C`: Regularization parameter (tested: 0.1, 1.0, 10.0, 100.0)

**Impact on Results:**
- **Low C (0.1)**: Creates wider margins, allows more misclassifications, simpler model, better generalization
- **High C (100.0)**: Creates narrower margins, fewer misclassifications, more complex model, risk of overfitting
- **Best for**: Linearly separable or nearly linearly separable data
- **Performance**: Fastest training, good baseline performance

**Key Finding**: Linear kernel works well when data is nearly linearly separable, but may underperform on complex non-linear datasets.

---

### 2. Polynomial Kernel
**Formula:** K(x, y) = (γ · x^T · y + coef0)^degree

**Parameters:**
- `C`: Regularization parameter (tested: 1.0, 10.0)
- `gamma`: Kernel coefficient (tested: 'scale', 'auto')
- `degree`: Polynomial degree (tested: 2, 3)

**Impact on Results:**
- **Degree 2**: Quadratic decision boundaries, moderate complexity
- **Degree 3**: Cubic decision boundaries, higher complexity, most common
- **Higher degrees**: More complex patterns but increased risk of overfitting
- **gamma='scale'**: Often performs better than 'auto' for feature-scaled data
- **Performance**: Computationally more expensive than linear or RBF

**Key Finding**: Polynomial kernel can capture non-linear relationships, but often outperformed by RBF kernel. Degree 3 with gamma='scale' typically provides best results.

---

### 3. RBF (Radial Basis Function) Kernel
**Formula:** K(x, y) = exp(-γ · ||x - y||²)

**Parameters:**
- `C`: Regularization parameter (tested: 0.1, 1.0, 10.0)
- `gamma`: Kernel coefficient (tested: 'scale', 'auto', 0.01, 0.1)

**Impact on Results:**
- **gamma='scale'**: Adaptive gamma based on feature variance (recommended, often best)
- **gamma='auto'**: Fixed gamma = 1/n_features (simpler but less optimal)
- **Low gamma (0.01)**: Wider influence radius, smoother boundaries, simpler model
- **High gamma (0.1)**: Narrower influence radius, more complex boundaries, risk of overfitting
- **C parameter**: Balances margin width and classification errors
- **Performance**: Most popular for non-linear problems, excellent generalization

**Key Finding**: RBF kernel typically achieves the best accuracy on non-linear datasets. The combination of C=1.0 and gamma='scale' is often optimal.

---

### 4. Sigmoid Kernel
**Formula:** K(x, y) = tanh(γ · x^T · y + coef0)

**Parameters:**
- `C`: Regularization parameter (tested: 1.0, 10.0)
- `gamma`: Kernel coefficient (tested: 'scale', 'auto', 0.01)

**Impact on Results:**
- **gamma='scale'**: Generally performs better than 'auto'
- **gamma='auto'**: May lead to poor performance
- **Custom gamma values**: Require careful tuning
- **Performance**: Often performs worse than RBF, less stable

**Key Finding**: Sigmoid kernel generally underperforms compared to RBF and requires careful parameter tuning. Not recommended as first choice.

---

## Parameter Effects Summary

### C Parameter (Regularization)
**Effect on All Kernels:**
- **Low C (0.1)**: 
  - Wider margin
  - More misclassifications allowed
  - Simpler model
  - Better generalization
  - Lower training accuracy, potentially higher test accuracy
  
- **High C (100.0)**:
  - Narrower margin
  - Fewer misclassifications
  - More complex model
  - Risk of overfitting
  - Higher training accuracy, potentially lower test accuracy

**Optimal Range**: Typically between 1.0 and 10.0 for most datasets.

### Gamma Parameter (RBF, Polynomial, Sigmoid)
**Effect:**
- **Low gamma (0.01)**:
  - Wider influence radius
  - Smoother decision boundaries
  - Simpler model
  - Better for large datasets
  
- **High gamma (0.1, 1.0)**:
  - Narrower influence radius
  - More complex boundaries
  - Risk of overfitting
  - Better for small datasets with complex patterns

**Recommendation**: Start with gamma='scale' (adaptive) rather than fixed values.

### Degree Parameter (Polynomial)
**Effect:**
- **Degree 2**: Quadratic boundaries, moderate complexity
- **Degree 3**: Cubic boundaries (most common), good balance
- **Higher degrees**: More complex but risk overfitting

**Recommendation**: Use degree 3 as default, avoid degrees > 5.

---

## Experimental Results Summary

### Dataset 1: Banknote Authentication (4 features, 1,372 samples)

**Best Performing Kernel**: RBF
- **Best Configuration**: C=1.0, gamma='scale'
- **Typical Accuracy**: ~99%+
- **Support Vectors**: Moderate number

**Observations**:
- Linear kernel performs well (data is nearly linearly separable)
- RBF kernel achieves slightly better accuracy
- Polynomial kernel competitive but slower
- Sigmoid kernel underperforms

### Dataset 2: Pima Indians Diabetes (8 features, 768 samples)

**Best Performing Kernel**: RBF
- **Best Configuration**: C=1.0, gamma='scale' or C=10.0, gamma='scale'
- **Typical Accuracy**: ~75-80%
- **Support Vectors**: Higher number (more complex boundary)

**Observations**:
- RBF kernel clearly outperforms others
- Linear kernel struggles with non-linear patterns
- Polynomial kernel shows moderate performance
- Sigmoid kernel requires extensive tuning

---

## Key Findings

### 1. Kernel Selection
- **RBF kernel** is the most versatile and typically performs best for non-linear problems
- **Linear kernel** works well for linearly separable data and serves as a good baseline
- **Polynomial kernel** can be competitive but is often outperformed by RBF
- **Sigmoid kernel** generally underperforms and requires careful tuning

### 2. Parameter Sensitivity
- **C parameter**: Moderate sensitivity; optimal values typically between 1.0 and 10.0
- **gamma parameter**: High sensitivity; 'scale' usually better than 'auto' or fixed values
- **degree parameter**: Moderate sensitivity; degree 3 is optimal for most cases

### 3. Feature Scaling
- **Critical**: SVM is highly sensitive to feature scales
- **Always use StandardScaler** before training
- Improves convergence and performance significantly
- Essential for kernel-based methods

### 4. Support Vectors
- Number of support vectors indicates model complexity
- Fewer support vectors = simpler, more generalizable model
- RBF often uses more support vectors than linear
- Can be used as complexity indicator

### 5. Overfitting Prevention
- Monitor train vs. test accuracy gap
- Lower C and gamma values reduce overfitting risk
- Cross-validation essential for parameter selection
- RBF with appropriate gamma is less prone to overfitting than high-degree polynomial

---

## Recommendations

### For New Datasets:

1. **Start with RBF kernel** with C=1.0 and gamma='scale' as default
2. **Try linear kernel** as baseline (fast and interpretable)
3. **Use grid search** for C and gamma optimization:
   - C: [0.1, 1, 10, 100]
   - gamma: ['scale', 'auto', 0.01, 0.1]
4. **Always scale features** before training (StandardScaler)
5. **Compare kernels** using cross-validation

### Parameter Tuning Workflow:

1. Scale features (StandardScaler)
2. Start with default parameters (C=1.0, gamma='scale')
3. Perform grid search: C ∈ [0.1, 1, 10, 100], gamma ∈ ['scale', 'auto', 0.01, 0.1]
4. Use cross-validation to select best parameters
5. Evaluate on test set

### When to Choose Each Kernel:

- **Linear**: Large datasets, linear problems, need for speed, interpretability
- **RBF**: Non-linear problems, unknown data distribution, general purpose (recommended default)
- **Polynomial**: Moderate non-linearity, need explicit complexity control
- **Sigmoid**: Specific use cases, experimental purposes (not recommended)

---

## Conclusion

Different kernel functions create different decision boundaries and have varying computational costs. The choice of kernel and parameters significantly affects classification performance. 

**Key Takeaways:**
1. **RBF kernel with C=1.0 and gamma='scale'** is the recommended starting point for most problems
2. **Feature scaling is essential** for SVM performance
3. **Parameter tuning through cross-validation** is crucial for optimal results
4. **Linear kernel** works well when data is nearly linearly separable
5. **Support vector count** can indicate model complexity and generalization potential

For the datasets analyzed:
- **Banknote Authentication**: Both linear and RBF perform excellently (~99%+ accuracy)
- **Pima Indians Diabetes**: RBF kernel clearly outperforms others (~75-80% accuracy)

The RBF kernel provides the best balance between accuracy and generalization for non-linear classification problems.

---

*This summary is based on experimental analysis conducted on Banknote Authentication and Pima Indians Diabetes datasets using scikit-learn's SVC implementation.*

# Evaluation Metrics
This section goes through popular evaluation metrics used for different ML and DL tasks

## Binary classification metrics
### 1. Accuracy
Accuracy measures the fraction of correctly predicted positive class samples out of all the samples in the test set.

$$ Accuracy = \frac{TP + TN}{TP + FP + TN + FN}$$

Important considerations:
\begin{itemize}
\item Do not use accuracy for imbalanced datasets. The majority class will dominate the metric.
\end{itemize}

### 2. Precision

### 3. Recall

### 4. F1 Score

### 5. AUC
AUC (Area Under Curve) measures the area under the Receiver Operating Characteristics (ROC) curve. 

#### ROC Curve
The ROC curve plots the FPR on x-axis and TPR on y-axis. 

$$ TPR = \frac{TP}{TP + FN} \n
 FPR = \frac{FP}{FP + TN} $$

Generally, binary classifications output a probability score which indicates the probability of the sample being positive. To convert these probabilities into a binary score, we often use a threshold. A score above that threshold value is a 1 and anythng below is a 0.
The ROC curve plots the TPR and FPR for different thresholds. This helps in determining a good threshold value to use, as well as gives an overall idea of how the model performs regardless of what threshold is used thus making it easy to compare different models overall without relying on thesholds (this is what AUC also does). 

The animation below gives a visual description of how the ROC curve is created using a confusion matrix.

<video width="600" controls>
  <source src="../images/roc.mp4" type="video/mp4">
</video>

The larger the area under the ROC curve, the better the model

### 6. AUCPR

## Multi-class classification metrics

## Regression metrics

## Ranking and recommendation metrics
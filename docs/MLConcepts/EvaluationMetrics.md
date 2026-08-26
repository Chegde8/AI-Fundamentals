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

- X-axis: $ FPR = \frac{FP}{FP + TN} $
- Y-axis: $ TPR (recall) = \frac{TP}{TP + FN} $


Generally, binary classifications output a probability score which indicates the probability of the sample being positive. To convert these probabilities into a binary score, we often use a threshold. A score above that threshold value is a 1 and anything below is a 0.
The ROC curve plots the TPR and FPR for different thresholds. This helps in determining a good threshold value to use, as well as gives an overall idea of how the model performs regardless of what threshold is used thus making it easy to compare different models overall without relying on thesholds (this is what AUC also does). 

The animation below gives a visual description of how the ROC curve is created using a confusion matrix.

<video width="600" controls>
  <source src="../../images/roc.mp4" type="video/mp4">
</video>

The larger the area under the ROC curve, the better the model

### 6. AUCPR
This is also a way to evaluate binary classification across all possible thresholds, like AUC.
However, instead of the ROC curve, the area under the precision-recall curve is computed. 

- X-axis: $ Recall (TPR) = \frac{TP}{TP + FN} $
- Y-axis: $ Precision = \frac{TP}{TP + FP} $

The key difference between ROC-AUC and PR-AUC is how they treat the negative class. 

- ROC-AUC's FPR term is normalized by the the total number of negatives (FP + TN). If negatives vastly outnumber positives, like in fraud detection, TN is huge, so even a large raw number of false positives barely moves FPR. This makes ROC-AUC look deceptively good even when a model is actually generating tons of false positives in absolute terms.

- PR-AUC's precision term is normalized by predicted positives (TP + FP). It never looks at TN. So it reflects "when I flag something as positive, how often am I right," which is what matters when false positives are costly or when you're operating in a highly imbalanced setting. Thus the PR-curve shows the balance between the precision-recall tradeoff which is generally what we care about in imbalanced datasets.

When to use which:
- PR-AUC: when positive class is rare (i.e. class imbalance), or when you care about precision for your use case.

- ROC-AUC: when classes are roughly balanced, or when you care about both classes equally (since it gives you a symmetric view of separability). 

## Multi-class classification metrics

## Regression metrics

## Ranking and recommendation metrics
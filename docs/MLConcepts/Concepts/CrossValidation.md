# Cross Validation

Cross-validation is a technique for estimating how well a model will generalize to unseen data. This is accomplished by systematically splitting the dataset into multiple train/validation combinations rather than relying on a single train/test split, which can give a misleadingly optimistic or pessimistic estimate depending on which examples happened to land in the test set.

### Why this matters

- A single train/test split has two problems: the performance estimate depends heavily on the luck of that particular split, and you're only evaluating on a fraction of the data. 

- Cross-validation addresses both:

* by rotating which subset acts as validation, **every example eventually gets evaluated on** rather than just a few which may be affected by outliers.

* You also get a **distribution of scores**, not just one number, which also lets you see **how stable your model's performance is across different subsets**: low variance across folds means a robust model, high variance is a red flag.

## Types of cross validation

### 1. K-fold cross validation
This is the standard approach: split the dataset into k equal-sized folds. For each of the k iterations, train on k-1 folds and validate on the remaining one fold, then rotate which fold is held out. At the end, you average the k validation scores to get your overall performance estimate.

- Larger k: each fold is smaller, however this means means more training data per round and lower bias, but higher variance and more compute, since you're training k separate models.

- Smaller k: each fold is larger, therefore larger validation set per round. Faster and lower variance, but each model sees slightly less training data, so slightly higher bias.

- The extreme case: k = number of samples. This is called **leave-one-out cross-validation (LOOCV)**, rarely used in practice except on very small datasets, since it's expensive and can have high variance.

Note: the deeper explanation for why variance increases as k grows (toward LOOCV): this is not due to just "smaller validation sets are noisier per fold." It's that with larger k, the training sets across different folds become highly overlapping with each other, e.g., with LOOCV, each training set differs from the next by only one example. This means the k models you train are highly correlated with each other, and averaging highly correlated estimates doesn't reduce variance as much as averaging independent ones would. That's why LOOCV, despite using almost all the data for training each time and having low bias, tends to have higher variance in its overall CV estimate than something like 5 or 10-fold.

### 2. Stratified k-fold
A variant of k-fold specifically for classification problems with **class imbalance**. Regular k-fold splits randomly, which means by chance one fold might end up with very few or even zero positive examples, especially with rare classes, giving you a wildly unreliable validation score for that fold.

Stratified k-fold fixes this by ensuring **each fold preserves the same class distribution as the overall dataset**. So if the full dataset is 2% fraud, every fold will also be approximately 2% fraud, rather than some folds accidentally getting 0.5% and others 5%. This gives much more stable, trustworthy validation scores when working with imbalanced data.

### 3. Time-aware (time series) split
Regular and stratified k-fold both assume examples are exchangeable, meaning shuffling them and randomly grouping them into folds is valid, order doesn't matter. But when there's a temporal structure, meaning future examples could leak information into predictions about the past, or more importantly, the model will only ever be deployed to predict the future in production, random folding creates look-ahead bias: you might train on data from March and validate on data from January, which the model would never realistically have access to in deployment.

Time-aware splitting instead respects chronological order. The most common version is **rolling-origin** or **expanding window** validation: train on data up through time t, validate on the period right after, then expand the training window and repeat, always keeping validation data strictly after training data in time. Sometimes a **fixed-size sliding window** is used instead, where the training window doesn't grow but slides forward, useful when older data may be less relevant, like older fraud patterns your model shouldn't over-rely on since fraud tactics evolve.
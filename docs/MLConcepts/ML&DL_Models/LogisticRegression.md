# Logistic Regression

Logistic regression is a classification algorithm, most commonly used for binary classification, that models the probability of an outcome belonging to a particular class, rather than predicting the class label directly. Despite the name containing "regression," it's fundamentally a classifier, the name comes from the fact that it's built on top of a linear regression-like structure before being transformed into a probability.

## Core idea

You start with a linear combination of your features, exactly like linear regression, this is often called the log-odds or logit:

$$z = {\beta}_0 + {\beta}_1x_1 + {\beta}_2x_2 + ... + {\beta}_nx_n$$

This $z$ can range from negative infinity to positive infinity, which isn't a valid probability. So logistic regression passes z through the sigmoid function to squash it into the range zero to one:

$$p = \frac{1}{(1 + e^(-z))}$$

This $p$ is interpreted as the predicted probability of the positive class. To get a hard classification, you apply a threshold, commonly 0.5, if $p$ is above the threshold, predict class 1, otherwise class 0. Importantly, that threshold is tunable, not fixed.

**Why sigmoid specifically**: The sigmoid function has a nice property, its inverse, the logit function, is exactly the log of the odds, meaning $log(\frac{p}{1-p}) = z$, the original linear combination. So logistic regression can be thought of as linear regression on the log-odds scale.

## How it is trained
Since the output is a probability, not a continuous unbounded value, logistic regression is trained using log loss, also called binary cross-entropy. Log loss heavily penalizes confident wrong predictions, predicting 0.99 probability for something that turns out to be class 0 incurs a huge penalty, this is what makes it well-suited to probability estimation rather than just classification accuracy.

## Interpretability
This is logistic regression's biggest advantage over more complex models. Each coefficient $\beta_i$ has a direct, clean interpretation: a one-unit increase in feature xᵢ multiplies the odds of the positive class by $e^(\beta_i)$, holding all other features constant. This is called the odds ratio. So if a coefficient for "company age in years" is 0.1, the odds ratio is e^0.1, roughly 1.105, meaning each additional year of company age is associated with about a 10.5% increase in the odds of being low-risk, holding everything else constant.

## Assumptions and limitations

- Assumes a linear relationship between features and the log-odds, not the raw probability, this is a common point of confusion, the probability-feature relationship is actually S-shaped, non-linear, but log-odds versus features is linear.

- Assumes no severe multicollinearity between features, correlated features destabilize coefficient estimates and interpretation.

- Struggles to capture feature interactions or non-linear patterns unless you manually engineer interaction terms or polynomial features, this is the main practical reason tree ensembles usually outperform it on complex tabular data, they capture interactions automatically via splits.

- Sensitive to outliers to some degree, though less so than linear regression, since the sigmoid compresses extreme values.

## When to use it

- When interpretability or regulatory explainability is a hard requirement.

- As a fast, lightweight baseline before trying more complex models, very standard practice.

- When you have a reasonably large dataset relative to feature count, and suspect roughly linear separability in log-odds space.

- When you need well-calibrated probabilities, logistic regression tends to produce more naturally calibrated probability outputs than tree ensembles, which often need explicit calibration, like Platt scaling or isotonic regression, to produce trustworthy probabilities. 

## Multi-class extension
Multinomial logistic regression, also called softmax regression, generalizes this to more than two classes, using the softmax function instead of sigmoid, this is the same idea underlying the final layer of most neural network classifiers.
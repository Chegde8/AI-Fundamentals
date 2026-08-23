# Confidence Intervals

A confidence interval (CI) gives you a range of plausible values for the true population parameter (e.g., the true difference in conversion rate between treatment and control), rather than just a single point estimate.

A 95% confidence interval means: if you repeated this experiment many times, and computed a CI each time using the same method, 95% of those intervals would contain the true population parameter.

It is NOT "there's a 95% probability the true value falls within this specific interval." Once the interval is computed from your one sample, the true value either is or isn't in it — there's no probability left, the randomness was in the sampling process, not the fixed true parameter.

** How CI connects to hypothesis testing **: if a 95% CI for the difference between treatment and control does not include 0, that's equivalent to rejecting H₀ at α = 0.05 in a two-tailed test. many experienced practitioners prefer reporting CIs over just p-values, because a CI shows both whether there's likely an effect (does it exclude 0) and how big it might be (effect size / practical significance), all in one number range.

#### Example
An e-commerce company runs an A/B test on a checkout page redesign. Here's the data:

- Control group: 10,000 users, 500 conversions --> Conversion rate = 5%

- Treatment group: 10,000 users, 500 conversions --> Conversion rate = 5.5%

We want a 95% confidence interval for the difference in conversion rates (treatment - control).

Formula for a CI on a difference in proportions:
$$ \hat{p_1} - \hat{p_2} \pm z^* x SE $$
Where:

- $\hat{p_1}$, $\hat{p_2}$ are the two sample proportions

- $z^*$ is the critical value for the confidence level (for 95%, $z^* = 1.96$)

- $ SE = sqrt(\frac{\hat{p_1}(1-\hat{p_1})}{n_1} + \frac{\hat{p_2}(1-\hat{p_2})}{n_2}) $ is the standard error of the difference

In this example,

- $\hat{p_treatment} = 0.055$, $n_treatment = 10,000$

- $\hat{p_control} = 0.050$, $n_control = 10,000$

- $z^* = 1.96$ for 95% confidence

Therefore,

- The point estimate of the difference $\hat{p_treatment} - \hat{p_control} = 0.005$

- The standard error $SE = 0.00315$

- The margin of error $z^* x SE = 0.006174$

- The full margin of error $= 0.005 \pm 0.006174 = (-0.001174, 0.011174) \approx (-0.12\%, 1.12\%)$

Interpreting the above:
The 95% CI for difference in conversion rate is approximately (-0.12\%, 1.12\%). **This interval includes 0.** Since 0 is a plausible value for the true difference (within our 95% confidence range), we cannot conclude there's a statistically significant difference between treatment and control at the 95% level, even though the observed point estimate (0.5 percentage points, a 10% relative lift) looks encouraging on the surface.

The CI-excludes-zero rule is equivalent to $p < 0.05$ in a two-tailed test. Since our interval straddles zero, if you ran the corresponding hypothesis test, you'd get $p > 0.05$ and fail to reject $H_0$.

** Why this matters practically: ** This is actually a very common and important real-world result: you observed a lift, but you don't have enough evidence (or enough sample size) to confidently say it's real, rather than noise. A 0.5 percentage point difference from 5.0% to 5.5% is a plausible real effect, but with 10,000 users per group and roughly 500 events, there's still enough sampling variability that the true difference could plausibly be slightly negative, zero, or positive.
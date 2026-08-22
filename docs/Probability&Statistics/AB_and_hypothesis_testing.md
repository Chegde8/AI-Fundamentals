# A/B Testing and Hypothesis Testing

## Hypothesis Testing
At its core, hypothesis testing is a framework for answering "Is what I am observing a real effect, or could it just be due to random chance?".

You start with two competing statements:

- **Null hypothesis ($H_0$):** the "boring" default - there is no effect, change, difference, or anything interesting going on. Eg: "the new checkout button color has no effect on conversion rate."

- **Alternative hypothesis ($H_1$ or $H_a$):** what you are actually trying to find evidence for - there is an effect, change, difference, or something interesting. Eg: "the new checkout button color changes conversion rate."


You never prove $H_0$ or $H_1$ as true. Instead, you either reject $H_0$ (evidence suggests something is going on) or fail to reject $H_0$ (not enough evidence to say something is going on. This is not the same as proving there's no effect.)


### Core mechanics
1. **Collect data** from 2 or more groups - E.g., control (old button) vs. treatment (new button)

2. **Compute a test statistic** - a number summarizing how different the observed data is from what you would expect under $H_0$ (e.g., a t-statistic for comparing means, z-statistic for comparing proportions).

3. **Compute a p-value** - given that $H_0$ is true, what is the probability of seeing a result at least as extreme as what you observed, just by random chance?

4. **Compare p-value to a significance threshold (\alpha)**, typically 0.05 - 
- $p-value < \alpha$ : reject $H_0$ (statistically significant result)
- $p-value \geq \alpha$ : fail to reject $H_0$ (not enough evidence)


## The p-value
The p-value is the probability of observing data at least as extreme as what you got, assuming the null hypothesis is true.

What it is not:

- It is NOT the probability of $H_0$ being true

- It is NOT the probability that your result is due to chance

- It is NOT the probability that $H_1$ is true

A p-value of 0.03 means: if there really were no effect, you'd see a result this extreme (or more extreme) only 3% of the time by random chance. It says nothing directly about how likely it is that there is an effect — that would require Bayesian reasoning with a prior, which frequentist p-values don't provide.

## Type I and Type II Errors
| | $H_0$ is actually true | $H_0$ is actually false |
| --- | --- | ---|
You reject $H_0$ | Type I Error (false positive) | Correct (true positive) |
| You fail to reject $H_0$ | Correct (true negative) | Type II Error (false negative) |

- Type I error rate = $\alpha$ - probability of concluding there's an effect when there isn't one.

- Type II error rate = $\beta$ - probability of missing a real effect that's actually there.

- Statistical power = $1 - \beta$ - probability of correctly detecting a real effect when it exists. To increase power you can increase sample size, increase effect size (harder to control), or reduce variance in your measurement.

There is a fundamental tradeoff: making \alpha smaller (stricter significance threshold) educes false positives but increases false negatives (lower power), all else equal.

## Designing an AB Test
1. **Define the metric** - pick a single primary metric that reflects the business goal (e.g., conversion rate), not multiple metrics you'll cherry pick from later.

2. **Define $H_0$ and $H_1$** - usually $H_0$ is that there is no difference in metric between control and treatment.

3. **Choose significance level $\alpha$** - typically 0.05, sometimes lower for high stakes decisions.

4. **Determine required sample size** before before running the test - based on:

    - Desired statistical power (commonly 80%) 

    - Minimum detectable effect (the smallest change you actually care about detecting)

    - Baseline conversion rate and variance

5. **Randomly assign users** to control and treatment - randomization is what allows you to attribute differences to the treatment causally, not to confounders.

6. **Run the test** for the predetermined duration / sample size - don't peek and stop early just because it looks significant. This inflates false positive rate, sometimes called peeking problem.

7. **Analyze results** - compute the test statistic, p-value, and often a confidence interval for the effect size (not just the significance, the magnitude of the effect matters for business decisions too).

8. **Make a decision** - reject or fail to reject $H_0$, but also weigh practical significance (is the effect big enough to matter for the business) vs. just statistical significance (is it likely a real, non-zero effect). A tiny but statistically significant effect might ot be worth shipping. 

### An example
Suppose a retail company wants to test whether adding "only 2 left in stock" urgency messaging increases conversion rate.

- $H_0$ - urgency messaging has no effect on conversion rate.

- $H_1$ - urgency messaging changes conversion rate. 

- Randomly split traffic 50/50 into control (no messaging) and treatment (messaging shown).

- Predetermine sample size needed to detect, say, a 1% absolute lift in conversion with 80% power at $\alpha$ = 0.05.

- Run for full predetermined period.

- Compute p-value comparing conversion rates between groups.

- If $p < 0.05$ and the lift is large enough to matter practically (e.g., not just 0.01% higher), ship it.

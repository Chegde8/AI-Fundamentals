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
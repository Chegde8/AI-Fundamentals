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

Rejecting $H_0$ doesn't mean "$H_0$ is probably not true" in a strict probability sense. That phrasing edges toward the p-value misinterpretation trap. The more precise statement is: "if $H_0$ were true, results this extreme would be rare — rare enough that we've decided (via our $\alpha$ threshold) to act as if H₀ is false." It's a decision rule, not a direct probability statement about $H_0$ itself.

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
1. **Define the metric** - pick a single primary metric that reflects the business goal (e.g., conversion rate). In practice, multiple metrics are monitored: 

    - One pre-registered primary metric — this is the one metric that determines whether the test "succeeded," and is decided before the test runs. This is what you use for the actual ship/no-ship decision.
    
    - Secondary/guardrail metrics — tracked alongside, to catch unintended side effects (e.g., "did revenue increase but customer complaints also spike?" or "did conversion go up but average order value went down?"). These inform the decision but aren't the primary pass/fail bar.

    - The reason to commit to one primary metric in advance is to prevent cherry-picking after the fact. If you test 10 metrics and don't pre-declare a primary one, you'll almost always find something that looks "significant" just by chance. Pre-registering the primary metric protects you from that trap, even while you're free to look at everything else for context and diagnosis.

2. **Define $H_0$ and $H_1$** - usually $H_0$ is that there is no difference in metric between control and treatment.

3. **Choose significance level $\alpha$** - typically 0.05, sometimes lower for high stakes decisions.

4. **Determine required sample size** before running the test - based on:

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


------ 

## Additional Information

#### One-tailed vs two-tailed testing

- Two-tailed test: $H_1$ is "there is a difference" (could be positive or negative). The p-value accounts for extremity in either direction. This is the default/safe choice, and what most A/B tests use, since you often can't rule out the treatment backfiring.

- One-tailed test: $H_1$ is directional — "the treatment increases the metric" (or specifically decreases it). The p-value only accounts for extremity in that one direction.

Practical implication: a one-tailed test has more statistical power to detect an effect in the direction you care about (for the same α), because all your "rejection region" is concentrated on one side of the distribution. But it comes with a real risk: if you use a one-tailed test expecting a positive effect, and the treatment actually makes things dramatically worse, your test won't flag that as "significant" in the way you'd want — because you specified the hypothesis to only look one direction.

In practice, most rigorous A/B testing frameworks default to two-tailed tests, since you generally do want to catch negative surprises, not just confirm positive ones. Choosing a one-tailed test to gain power is sometimes viewed skeptically, since it can look like you're trying to make it easier to get a "significant" result in your preferred direction — worth being aware of as a potential ethical/rigor gray area.

#### Effect size
Effect size measures the magnitude of a difference or relationship, independent of sample size or statistical significance. It answers "how big is the effect," as opposed to the p-value, which answers "how confident are we that an effect exists at all."

Why this distinction matters: with a large enough sample size, even a tiny, practically meaningless difference can become statistically significant ($p < 0.05$), because p-values are heavily influenced by sample size. Effect size tells you whether that difference is actually big enough to matter for the business.

Common effect size measures:

- Cohen's d (for comparing two means): standardized difference between two group means, in units of standard deviation. Rules of thumb: 0.2 = small, 0.5 = medium, 0.8 = large.

- Absolute or relative lift (very common in A/B testing/business contexts): e.g., "conversion rate went from 5% to 5.3%" — an absolute lift of 0.3 percentage points, or a relative lift of 6%.

- Correlation coefficient (r) — itself a kind of effect size for relationships between variables.

#### What goes wrong without a precomputed sample size

- Underpowered test → false negatives. If your sample size is too small for the effect size you're hoping to detect, you might genuinely have a real effect happening, but your test lacks the statistical power to detect it reliably — you'd likely fail to reject H₀ and wrongly conclude "no effect," when really you just didn't collect enough data to see it clearly.

- "Peeking" problem → false positives. This is the sneakier one, and it's what happens when people run a test without a predetermined sample size/duration and instead check results continuously, stopping as soon as it "looks significant." Since p-values fluctuate over time as data accumulates, if you keep checking and stop the moment p < 0.05, you dramatically inflate your true false-positive rate — even if each individual look seems legitimate. This is closely related to the multiple testing problem (repeated looks = repeated tests). The fix is to commit to a sample size/duration in advance (or use specialized "sequential testing" methods designed to allow valid peeking, which is a more advanced topic).

#### Multiple testing problem
When you run many statistical tests simultaneously (or repeatedly), the probability that at least one of them shows a "significant" result purely by chance increases, even if there's no real effect anywhere.

Concrete intuition: if $\alpha = 0.05$, there's a 5% chance any single test gives a false positive. If you run 20 independent tests (e.g., testing 20 different metrics, or 20 different customer segments), the probability that at least one comes back "significant" purely by chance is:
$$ 1 - ( 1- 0.05)**20 \approx 64\% $$
That's a huge inflation from the intended 5% error rate, and it's why "we tested 20 metrics and found one with $p=0.04$" is a red flag instead of being a win.

Common corrections:

- Bonferroni correction: divide $\alpha$ by the number of tests (e.g., $\alpha/20 = 0.0025$ per test) — simple but conservative (increases Type II error/false negatives).

- Benjamini-Hochberg (False Discovery Rate control): a less conservative alternative, more commonly used when running many tests (e.g., genomics, large-scale experimentation platforms) since Bonferroni becomes overly strict as the number of tests grows.

This is why preregistering one primary metric is important - it sidesteps the multiple testing problem entirely for the main ship decision, even while you still look at secondary metrics more cautiously/diagnostically.

#### Does the value of p-value matter:
For the actual decision (ship / don't ship, reject / fail to reject $H_0$), it's binary. You pick $\alpha$ in advance, and the decision rule is $p < \alpha$ or not. Whether p = 0.049 or p = 0.001, both "reject $H_0$" — the decision itself doesn't get "more true" as p gets smaller.

But the magnitude of the p-value isn't entirely thrown away in practice, for a few reasons worth knowing:

- Strength of evidence, informally: a p-value of 0.001 is often talked about informally as "stronger evidence against $H_0$" than $p = 0.049$, even though both cross the same threshold. This isn't a formal probability statement (it's still not "how likely $H_0$ is false"), but analysts do use it as a rough signal of how convincingly the data deviates from the null.

- Sensitivity to the threshold itself: a result at $p = 0.049$ vs. $p = 0.051$ straddles the $α = 0.05$ line, but the actual strength of evidence is nearly identical. Treating one as "significant" and the other as "not" is a somewhat arbitrary artifact of the threshold, not a meaningfully different outcome. Good analysts are aware of this and don't treat 0.05 as some sharp, magical line. A result just barely missing significance is often treated as "suggestive, worth another look," not "definitely nothing there."

- Multiple testing corrections directly use the magnitude — Bonferroni and similar corrections adjust the $\alpha$ threshold based on how many tests you're running, so the actual p-value magnitude matters for whether it clears the corrected bar, even if the underlying logic per test is still binary.
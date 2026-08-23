# KL Divergence
Kullback Leibler Divergence is a measure that quantifies difference between two probability distributions.
It tells:  
* How much information is lost when we approximate a true distribution P with another distribution Q.  
* It is also called relative entropy and is non-negative and asymmetric, i.e. D_KL(P||Q) != D_KL(Q||P).  

## Formula for KL divergence 
Between two probability distributions $P = \{p_1, p_2, ..., p_n\}$ and $Q = \{q_1, q_2, ..., q_n\}$  
For discrete distributions:  
$$D_{KL}(P||Q) = \sum_{i=1}^{n}p_ilog\frac{p_i}{q_i}$$

For continuous distributions:  
$$D_{KL}(P||Q) = \int p(x)log\frac{p(x)}{q(x)}dx$$
$p(x)$ and $q(x)$ are continuous probability density functions.

## Properties
1. Non-negativity: KL divergence is always non-negative. It is equal to 0 if P=Q.
2. Asymmetric: KL divergence is not symmetric and therefore is not a true distance metric. $D_{KL}(P||Q) \neq D_{KL}(Q||P)$
3. Additivity for independent distributions: If X and Y are independent, $D_{KL}(P_{X,Y}||Q_{X,Y}) = D_{KL}(P_X||Q_X) + D_{KL}(P_Y||Q_Y)$
4. Expectation form: It can be interpreted as the expected logarithmic difference between probabilities under P and Q, $D_{KL}(P||Q) = E_{x~P}[log\frac{P(x)}{Q(x)}]$
5. Minimizing KL divergence is the same as minimizing cross-entropy loss when some conditions are met.

## References
1. [Geeks for Geeks KL Divergence](https://www.geeksforgeeks.org/machine-learning/kullback-leibler-divergence/)
2. [Stackoverflow difference between KL divergence and cross entropy](https://stats.stackexchange.com/questions/357963/what-is-the-difference-between-cross-entropy-and-kl-divergence)

   

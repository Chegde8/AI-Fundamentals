# Informer
Informer tackles one of the most persistent headaches in time-series forecasting: the quadratic computation and memory requirements of 
standard Transformers when dealing with long sequences. It is designed specifically for long-sequence time series forecasting. It's goal is to: 
handle very long input sequences, reduce huge computational and memory costs of traditional transformers, improve stability and accuracy for 
multi-step or long-horizon forecasting.  

![Informer model](https://github.com/Chegde8/AI-Fundamentals/blob/main/ModelArchitectures/Transformers/images/informer.png?raw=true "Title")

![Single stack in informer's encoder](https://github.com/Chegde8/AI-Fundamentals/blob/main/ModelArchitectures/Transformers/images/informer_single_stack.png?raw=true "Title")

These components are what make Informers different from traditional transformers:  
1. ProbSparse self-attention: allows Informer to focus on the most important connections in the data, dramatically reducing computational complexity.
   * Self-attention is O(n^2) in time and memory. On a sequence of length n, it computes attention on all n x n pairs of tokens. This is
     impossible for long time series.
   * With ProbSparse attention, only the most informative queries contribute significantly to attention. It selects top-u queries based on
     KL divergence scoring and ignores the rest.
   * This reduces complexity from O(n^2) to O(nlogn) with no major loss in accuracy.
   * This is the main reason why Informer is used for long time series forecasting.
3. Self-attention distilling (downsampling): By progressively refining the input sequence, Informer can extract the essence of the data without getting bogged
  down in details.
   * In traditional transformers, all encoder layers operate on full length of sequence. This results in high computational and memory use and 
   risks overfitting when sequences are long.
   * In self-attention distilling, after each attention block, the sequence is downsampled (like pooling).
   * This keeps the most important timesteps, removes noisy and redundant points, and shrinks sequence layer by layer. 
5. Generative style decoder for long-horizon prediction: Unlike traditional decoders that predict step-by-step, Informer’s decoder can generate long output sequences in one go.
   * Traditional transformer decoders are autoregressive (i.e. predict next token and feed it back). This is expensive for multistep forecasting 
   and slow for long prediction windows.
   * The informer decoder uses a generative-style decoding mechanism that predicts many future steps in parallel, is not autoregressive, and thus 
   is much faster for multi-step horizons

## References
1. [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/pdf/2012.07436)
2. [Informer Revolutionizing Time-Series Forecasting](https://medium.com/@bijit211987/transformers-like-informer-arrevolutionizing-time-series-forecasting-f4e4ebd7db1b)

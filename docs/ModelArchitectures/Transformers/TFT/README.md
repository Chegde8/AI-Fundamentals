# Temporal Fusion Transformers
Temporal fusion transformer (TFT) is a model used for multi-horizon and multivariate time series forecasting.  

![Temporal fusion transformer](https://github.com/Chegde8/AI-Fundamentals/blob/main/ModelArchitectures/Transformers/images/tft.png?raw=true "Title")

A basic transformer, like the ones used in NLP, are mainly designed for sequence-to-sequence tasks, processing token embeddings, using multi-head attention for contextual understanding.  
But time series forecasting has unique needs:
* known future inputs (eg: holidays, prices)  
* static features (like store location, product type)  
* variable input importance  
* multi-horizon outputs  
* interpretability  
A basic transformer does not handle these things natively.

The key differences between TFT and a basic transformer are:  
1. TFT handles multiple types of inputs
   * Static features (store ID, product category, etc)  
   * Observed past features (sales, weather history)
   * Known future features (holidays, price plans, promotions)  
A basic transformer only handles sequences. It has no native mechanism for static or future-known inputs.  
2. TFT uses gating layers to select important inputs. It adds:  
   * Variable selection networks (VSN) which learn which features matter at each timestep.
   * Gated residual networks (GRN) which let the model skip irrelevant transformations.
   * Temporal gating which controls the flow of information over time.  
Transformers do not select features dynamically. They treat all input tokens equally.
3. TFT combines LSTMs + transformers. TFT has:
   * LSTM encoders for local/short-term patterns
   * Self-attention layers for long-term dependencies.  
This hybrid architecture is crucial for time series, where both short and long range patterns matter. A basic transformer does all this modeling through self-attention with no recurrent components.
4. TFT is designed for multi-horizon forecasting
   * TFT predicts multiple future timesteps at once. Eg, next 7 days, 30 days, 12 months.  
Basic transformers often focus on next token or sequence-to-sequence prediction, not structured multi-step forecasting.
5. Interpretability built in. TFT provides:
   * Variable importance (which feature matters)
   * Attention interpretability
   * Temporal importance  
Basic transformers are not inherently interpretable.
6. Handles heterogeneous time series. TFT is built to scale across:  
   * thousands of products
   * hundreds of locations
   * multiple time series  
It does feature gating and variable selection for each individual sequence. Basic transformers treat the entire input as one token sequence.

## References
1. [Understanding temporal fusion transformers](https://medium.com/dataness-ai/understanding-temporal-fusion-transformer-9a7a4fcde74b)
2. [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)

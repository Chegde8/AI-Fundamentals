# 1D CNNs
1D convolutional neural networks are designed to process sequential data, such as time-series signals, text, or audio data. It performs convolution operation on one-dimensional data.
The kernel slides along a single spatial dimension. 

The shape of input data is (batch_size, sequence_length, num_features)

![1D CNN](https://github.com/Chegde8/AI-Fundamentals/blob/main/ModelArchitectures/CNN/images/1DCNN_architecture.png?raw=true "Title")

## How it works
1. A kernel (filter) which is a small matrix of learnable weights is used to extract features from the sequence.
2. The kernel slides over the input sequence, performing a dot product at each position.
3. The result of this operation is a transformed feature map, capturing patterns in data.

## Advantages
1. 1D CNNs are useful to extract local patterns from sequential data, for example: peaks and valleys in time-series data, word or phrase relationships in text, short term features in audio signals.
2. Efficient for sequential data with short term dependencies: captures local dependencies effeciently. Specifically, dependencies within the the kernel window are captured well.
3. Parameter sharing: reduces number of parameters compared to fully connected networks.
4. Adaptable: can be combined with pooling layers or stacked for deeper architectures.

## Limitations
1. Short-term dependencies: may miss longer term dependencies without additional layers like RNNs.
2. Data format: requires data to be preprocessed into suitable shape.

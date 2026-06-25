BTC Price Transformer

This project uses a Transformer-based neural network to predict Bitcoin price movements using historical OHLC (Open, High, Low, Close) data.

The model is trained to predict the next closing price based on a sequence of previous 30 time steps.

Overview

The goal of this project is to forecast short-term Bitcoin price behavior using deep learning.

The model learns patterns from historical market data and attempts to predict the next closing price.

Dataset

The dataset consists of historical Bitcoin price data with the following features:
- Open price
- High price
- Low price
- Close price
- Date

The data is sorted chronologically before training.

Preprocessing

- Data is converted into sliding windows of 30 time steps
- Each sample contains 30 consecutive OHLC entries
- The target is the closing price of the next time step
- Data is normalized using training set mean and standard deviation

Model Architecture

The model is a Transformer encoder designed for time-series regression.

It includes:
- Linear projection from 4 features to embedding space
- Learnable positional embeddings
- Transformer encoder stack
- Final linear layer for regression output

Training

The model is trained using:
- Mean Squared Error loss function
- Adam optimizer

Training is performed on 80% of the dataset, with 20% reserved for testing.

Evaluation

The model is evaluated using:
- Mean Squared Error on normalized values
- Mean Absolute Error in real price scale

Predictions are compared against actual Bitcoin prices to measure error.

Limitations

This model does not guarantee accurate financial predictions.

It may be affected by:
- Market randomness and volatility
- Non-stationary time series behavior
- Overfitting on historical patterns
- Lack of external market signals (news, volume, sentiment)

This project is for educational purposes only and should not be used for financial decisions.

# 🌱 Soil Moisture Prediction with LSTM

This project focuses on predicting soil moisture conditions using machine learning and deep learning techniques. It leverages time-series data, preprocessing, and an **LSTM (Long Short-Term Memory) neural network** for classification.

## 📌 Features
- Data preprocessing with **pandas** and **scikit-learn**  
- Data visualization with **matplotlib** and **seaborn**  
- Sequence generation for time-series modeling  
- LSTM-based neural network using **TensorFlow/Keras**  
- Model evaluation using **accuracy, confusion matrix, and classification report**

## ⚙️ Requirements
Make sure you have the following installed:

```bash
python >= 3.8
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow
keras
```
## 📊 Model

The LSTM model architecture:
- Two LSTM layers with 50 units each
- Dropout regularization
- Dense output layer with sigmoid activation
- Optimizer: Adam
- Loss: Binary Crossentropy

## 📈 Results

The notebook includes:
- Accuracy score
- Confusion matrix heatmap
- Classification report

## 🌍 Applications
- Smart irrigation systems
- Precision agriculture
- Drought monitoring
- Environmental sustainability

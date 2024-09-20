import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from utils.preprocess import preprocess_data

# Load dataset
dataset = np.load('data/public_data.npz', allow_pickle=True)
X, y = dataset['data'], dataset['labels']

# Preprocess data. Use as test the entire dataset
X_test, y_test = preprocess_data(X, y)

# Load the saved model
model = tfk.models.load_model('models/AN2DL2023Challenge1Polimi.keras')

# Predict
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).flatten()

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Optionally, plot confusion matrix
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

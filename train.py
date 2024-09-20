import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from utils.preprocess import preprocess_data

# Set seed
seed = 2

# Load dataset
dataset = np.load('data/public_data.npz', allow_pickle=True)
X, y = dataset['data'], dataset['labels']

# Preprocess data
X_clean, y_clean = preprocess_data(X, y)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_clean, y_clean, random_state=seed, test_size=.20)

# Model definition
base_model = tf.keras.applications.ConvNeXtBase(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model.trainable = True

N = 205
for i, layer in enumerate(base_model.layers[:N]):
  layer.trainable=False
for i, layer in enumerate(base_model.layers):
  print(i, layer.name, ", trainable: ",layer.trainable)

model = tfk.Sequential([
        tfkl.RandomZoom(0.3),
        tfkl.RandomFlip('vertical'),
        tfkl.RandomContrast(0.3),
        tfkl.RandomRotation(0.3, fill_mode="reflect", interpolation="bilinear"),
        base_model,
        tfkl.Flatten(),
        tfkl.Dense(2048, activation='leaky_relu', kernel_initializer=xavier_init, kernel_regularizer = ridge),
        tfkl.Dropout(0.3),
        tfkl.BatchNormalization(),
        tfkl.Dense(1024, activation='leaky_relu', kernel_initializer=xavier_init, kernel_regularizer = ridge),
        tfkl.Dropout(0.3),
        tfkl.BatchNormalization(),
        tfkl.Dense(256, activation='leaky_relu', kernel_initializer=xavier_init, kernel_regularizer = ridge),
        tfkl.Dropout(0.3),
        tfkl.BatchNormalization(),
        tfkl.Dense(64, activation='relu', kernel_initializer=xavier_init, kernel_regularizer = ridge),
        tfkl.Dense(1, activation='sigmoid', kernel_initializer=xavier_init)
    ])

model.compile(loss=tfk.losses.BinaryCrossentropy(), optimizer='Adamax', metrics=['accuracy', 'Precision','Recall'])

# Compute class weights to balance the train set
class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y =y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))

# Callbacks
lr_scheduler = tfk.callbacks.ReduceLROnPlateau(
  monitor='val_accuracy',
  patience=5,
  factor=0.999,
  mode='max',
  min_lr=1e-5
)

early_stopping = tfk.callbacks.EarlyStopping(
  monitor='val_accuracy',
  mode='max',
  patience=10,
  restore_best_weights=True
)

# Train model
history = model.fit(
  X_train,
  y_train,
  batch_size=32,
  epochs=200,
  validation_data=(X_val, y_val),
  callbacks=[early_stopping, lr_scheduler],
  class_weight=class_weights
)

# Save model
model.save('models/AN2DL2023Challenge1Polimi.keras')

# Predict labels for the entire test set
predictions = (model.predict(X_val)>0.5).astype(int)

# Compute classification metrics
accuracy = accuracy_score(y_val, predictions)
precision = precision_score(y_val, predictions, average='macro')
recall = recall_score(y_val, predictions, average='macro')
f1 = f1_score(y_val, predictions, average='macro')

# Display the computed metrics
print('Accuracy:', accuracy.round(4))
print('Precision:', precision.round(4))
print('Recall:', recall.round(4))
print('F1:', f1.round(4))

# Plot training history
plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'], alpha=.3, color='#ff7f0e', linestyle='--')
plt.plot(history.history['val_loss'], label='Val loss', alpha=.8, color='#ff7f0e')
plt.legend(loc='upper left')
plt.title('Binary Crossentropy')
plt.grid(alpha=.3)

plt.figure(figsize=(15, 5))
plt.plot(history.history['accuracy'], alpha=.3, color='#ff7f0e', linestyle='--')
plt.plot(history.history['val_accuracy'], label='Val accuracy', alpha=.8, color='#ff7f0e')
plt.legend(loc='upper left')
plt.title('Accuracy')
plt.grid(alpha=.3)

plt.show()

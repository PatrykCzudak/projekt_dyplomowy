import numpy as np
import tensorflow as tf
import pickle

# Load data
X_other = np.load('X_other.npy')
X_symbol = np.load('X_symbol.npy')
y = np.load('y.npy')

SEQ_LEN = X_other.shape[1]
FEATURES = X_other.shape[2]

# Split
split_idx = int(len(X_other) * 0.8)
X_other_train, X_other_test = X_other[:split_idx], X_other[split_idx:]
X_symbol_train, X_symbol_test = X_symbol[:split_idx], X_symbol[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Model
input_other = tf.keras.Input(shape=(SEQ_LEN, FEATURES), name='other_features')
input_symbol = tf.keras.Input(shape=(SEQ_LEN, 1), name='symbol_code')

symbol_embedding = tf.keras.layers.Embedding(
    input_dim=np.max(X_symbol)+1, output_dim=16
)(input_symbol)
symbol_embedding = tf.keras.layers.Reshape((SEQ_LEN, 16))(symbol_embedding)

x = tf.keras.layers.Concatenate()([input_other, symbol_embedding])

x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='causal', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='causal', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

attention_output = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
attention_output = tf.keras.layers.Add()([x, attention_output])
attention_output = tf.keras.layers.LayerNormalization()(attention_output)
x = tf.keras.layers.Dense(128, activation='relu')(attention_output)
x = tf.keras.layers.GlobalAveragePooling1D()(x)

x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

output = tf.keras.layers.Dense(1, activation='tanh')(x)

model = tf.keras.Model(inputs=[input_other, input_symbol], outputs=output)
model.compile(optimizer='adam', loss=tf.keras.losses.Huber())
model.summary()

# Train
history = model.fit(
    {'other_features': X_other_train, 'symbol_code': X_symbol_train},
    y_train,
    validation_data=(
        {'other_features': X_other_test, 'symbol_code': X_symbol_test}, y_test
    ),
    epochs=30,
    batch_size=128
)

# Save
model.save('prediction_model_5d.h5')
print("[INFO] Model zapisany.")

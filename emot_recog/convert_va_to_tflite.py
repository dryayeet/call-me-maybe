import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

weights_path = 'models/regressor_weights.h5'
tflite_path = 'models/vaModel.tflite'

# Reconstruct architecture
base = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base.output)
x = Dense(1024, activation='relu')(x)
x = Dense(2, activation='linear')(x)
model = Model(inputs=base.input, outputs=x)
model.load_weights(weights_path)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

h5_size = os.path.getsize(weights_path) / 1024
tflite_size = os.path.getsize(tflite_path) / 1024
print(f'H5 weights: {h5_size:.1f} KB')
print(f'TFLite:     {tflite_size:.1f} KB')
print(f'Saved to {tflite_path}')

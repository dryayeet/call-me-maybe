import os
import tensorflow as tf

hdf5_path = 'models/emotionModel.hdf5'
tflite_path = 'models/emotionModel.tflite'

model = tf.keras.models.load_model(hdf5_path, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

hdf5_size = os.path.getsize(hdf5_path) / 1024
tflite_size = os.path.getsize(tflite_path) / 1024
print(f'HDF5:   {hdf5_size:.1f} KB')
print(f'TFLite: {tflite_size:.1f} KB')
print(f'Saved to {tflite_path}')

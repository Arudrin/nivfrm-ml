import tensorflow as tf


def convert_to_lite(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(path + '/' + 'segmentation.tflite', 'wb') as file:
        file.write(tflite_model)

import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("alzheimer_model.h5")

img = tf.keras.preprocessing.image.load_img(
    "test.jpg", target_size=(128,128)
)

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

prediction = model.predict(img_array)

classes = ['Mild', 'Moderate', 'Non', 'VeryMild']

print("Prediction:", classes[np.argmax(prediction)])
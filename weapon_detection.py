# importing libraries
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import time

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set_memort_growth TRUE
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

a = time.time()

# Preprocess train set
train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
    'weapon_classification/training', target_size=(64, 64), batch_size=32, class_mode='binary')


# Preporcess test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'weapon_classification/test', target_size=(64, 64), batch_size=32, class_mode='binary')


# Initialize CNN
cnn = tf.keras.models.Sequential()

# Step 1 Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
        activation='relu', input_shape=[64, 64, 3]))

# Step 2 Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

# Add second convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full Connection
cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))

# Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Train CNN
# Compile CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN on training set and Evaluate on test set
cnn.fit(x=training_set, validation_data=test_set, epochs=1)


# making prediction
test_image = image.load_img(
    'weapon_classification/predict/4.jpg', target_size=(64, 64))
test_image.show()    
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'weapon'
else:
    prediction = 'no_weapon'

print(prediction)


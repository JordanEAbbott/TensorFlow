import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = os.path.join('/Users/jorda/Documents/Year 2 Work/tensorflow/rock-paper-scissors/rps/')
VALIDATION_DIR = os.path.join('/Users/jorda/Documents/Year 2 Work/tensorflow/rock-paper-scissors/rps-test-set/')

training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)
training_data = training_datagen.flow_from_directory(TRAINING_DIR,
                                                     target_size=(150, 150),
                                                     batch_size=126,
                                                     class_mode='categorical')
validation_data = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                         target_size=(150, 150),
                                                         batch_size=126,
                                                         class_mode='categorical')

def train_rps(epochs_num):
    
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy')>0.99):
                print("\nReached 99% accuracy so stopping training.")
                self.model.stop_training = True
                
    callbacks = myCallback()
    
    model = tf.keras.models.Sequential([
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')])
    
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    history = model.fit(training_data, epochs=epochs_num, steps_per_epoch=20, validation_data=validation_data, verbose=1, validation_steps=3, callbacks=[callbacks])
    
    return model, history

def model_history(history):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    
    plt.show()
    
def predict_rps(model, img_path):
    
    path = os.path.join(img_path)
    
    img = image.load_img(path, target_size=(150, 150))
    img = np.array()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    plt.imshow(img)
    
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes)
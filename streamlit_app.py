import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import model_from_json

def generate_model():
    # Set the path to your dataset directory
    dataset_dir = 'C:\\Users\\Tulyahikayo Tevin\\Desktop\\TEVINS DOGS'

    # Define the input shape of the images and the number of classes (dog breeds)
    input_shape = (200, 200, 3)
    num_classes = 3

    # Set the batch size and number of epochs for training
    batch_size = 32
    epochs = 20

    # Create an instance of the ImageDataGenerator for data augmentation and preprocessing
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalize pixel values between 0 and 1
        validation_split=0.2  # Split the dataset into training and validation sets
    )

    # Load the training set images
    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Load the validation set images
    validation_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Create the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(200, 200, 3)),
        tf.keras.layers.Dense(2048, activation=tf.nn.relu),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Set the number of steps per epoch
    train_steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size

    # Define a learning rate schedule (optional)
    def schedule(epoch, lr):
        if epoch < 50:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    lr_scheduler = LearningRateScheduler(schedule)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[lr_scheduler]
    )

    # Save the model as .json and .h5 files
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")

    # Optionally, you can also save the model architecture and weights separately
    model.save("model_full.h5")  # Save the model architecture and weights together

    

# Streamlit app code


import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your data directories
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# Define hyperparameters
input_shape = (224, 224, 3)
num_classes = 2
batch_size = 32
epochs = 10

# Define the MnasNet architecture
def create_mnasnet_model(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    base_model = tf.keras.applications.MnasNet(input_tensor=input_tensor, weights=None, include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output)
    return model

# Create the model
model = create_mnasnet_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation for the training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# No data augmentation for validation data
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Save the trained model
model.save('mnasnet_binary_classification.h5')

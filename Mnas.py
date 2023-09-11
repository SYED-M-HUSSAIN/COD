import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Define MnasNet architecture
def mnasnet_model(input_shape=(224, 224, 3), num_classes=2):
    input_tensor = Input(shape=input_shape)
    
    # Stem
    x = Conv2D(32, kernel_size=3, strides=2, padding='valid')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Depthwise Separable Conv Blocks
    for filters, num_blocks, stride in [(32, 3, 1), (64, 4, 2), (128, 2, 2), (256, 2, 2)]:
        for _ in range(num_blocks):
            # Depthwise Convolution
            x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            
            # Pointwise Convolution
            x = Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Classifier
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

# Create the MnasNet model
model = mnasnet_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation and loading
batch_size = 32
image_size = (224, 224)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'data_dir',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data_dir',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Save the trained model
model.save('mnasnet_binary_classification.h5')

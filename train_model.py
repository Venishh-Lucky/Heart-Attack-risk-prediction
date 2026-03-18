import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import os

# 1. Configuration
DATASET_DIR = 'path/to/eyepacs_dataset' # You will update this
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = 'heart_model.h5'

def build_model():
    print("Building model with ResNet50 base...")
    # Using Transfer Learning: ResNet50 pre-trained on ImageNet
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model to keep pre-trained patterns
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Probability output (0 to 1)
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} not found.")
        print("Please download EyePACS from Kaggle and update the DATASET_DIR path.")
        return

    model = build_model()

    # Data Augmentation (Crucial for high accuracy)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    print(f"Training complete. Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()

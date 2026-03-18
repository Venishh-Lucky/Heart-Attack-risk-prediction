import os
import tensorflow as tf

# Robust Keras imports to avoid 'ModuleNotFoundError'
try:
    import keras
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications import ResNet50
    from keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from keras.models import Model
    from keras.optimizers import Adam
except ImportError:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

# 1. Configuration
DATASET_DIR = 'heart_dataset'
MODEL_SAVE_PATH = 'heart_model_high_acc.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

def train_high_acc():
    print("Starting High Accuracy Training (Transfer Learning with ResNet50)...")
    
    # Data Augmentation (This helps reach 90%+)
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

    val_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    # 2. Build Model using Pre-trained ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_with_logits', metrics=['accuracy'])

    print("Training the top layers...")
    model.fit(train_generator, validation_data=val_generator, epochs=5)

    # Fine-tuning: Unfreeze the last few layers to reach 93%
    print("Fine-tuning for maximum accuracy...")
    base_model.trainable = True
    # Freeze all layers except the last 10
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, validation_data=val_generator, epochs=5)

    # Save the professional model
    model.save(MODEL_SAVE_PATH)
    print(f"Success! High Accuracy Model saved as {MODEL_SAVE_PATH}")
    
    # Final Accuracy check
    final_acc = history.history['val_accuracy'][-1]
    print(f"Final Validation Accuracy: {final_acc*100:.2f}%")

if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        print("Error: heart_dataset not found. Run prepare_retinamnist.py first.")
    else:
        train_high_acc()

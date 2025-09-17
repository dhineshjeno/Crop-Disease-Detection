import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def create_model(num_classes, img_size=128):
    """Create an improved CNN model for plant disease classification"""
    
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_plant_disease_model():
    """Train the plant disease classification model"""
    
    # =========================
    # Optimized Configuration
    # =========================
    IMG_SIZE = 64       # small image size to save RAM
    BATCH_SIZE = 10  # increased from 4, safe for 16GB RAM
    EPOCHS = 8       # increased from 3 for better training
    
    # Check if dataset exists
    dataset_path = "data/PlantVillage"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run download_dataset.py first")
        return False
    
    print("Setting up data generators...")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"Found {train_generator.samples} training images")
    print(f"Found {validation_generator.samples} validation images")
    print(f"Number of classes: {train_generator.num_classes}")
    
    # Save class labels
    class_labels = list(train_generator.class_indices.keys())
    with open("class_labels.json", "w") as f:
        json.dump(class_labels, f, indent=2)
    
    print("Class labels saved to class_labels.json")
    
    # Create model
    print("Creating model...")
    model = create_model(train_generator.num_classes, IMG_SIZE)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001
        )
    ]
    
    # Calculate class weights for imbalanced dataset
    y_train = train_generator.classes
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print("Starting training...")
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # Save model
    model.save("plant_disease_model.h5")
    print("Model saved as plant_disease_model.h5")
    
    # Save training history
    with open("training_history.json", "w") as f:
        json.dump(history.history, f, indent=2)
    
    # Evaluate model
    print("Evaluating model...")
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    return True

if __name__ == "__main__":
    print("Starting Plant Disease Model Training...")
    success = train_plant_disease_model()
    if success:
        print("Training completed successfully!")
        print("You can now run the FastAPI server with: uvicorn api:app --reload")
    else:
        print("Training failed. Please check the errors above.")

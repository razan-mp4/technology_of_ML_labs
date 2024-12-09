import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Parameters for training
img_size = 128  # The size of the images to which all dataset images will be resized
batch_size = 32  # The number of images processed in each batch during training
initial_epochs = 5  # The number of training epochs for the initial training
overfit_epochs = 15  # Number of epochs for demonstrating overfitting

# Data Generators for loading and preprocessing data
# Train Generator: Loads training images and applies rescaling (normalizing pixel values between 0 and 1)
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    'lab4/train',  # Path to the training dataset
    target_size=(img_size, img_size),  # Resize images to the specified size
    batch_size=batch_size,  # Number of images in each batch
    class_mode='binary',  # Binary classification (cats vs dogs)
    subset='training'  # Use 80% of the training data as training set
)

# Validation Generator: Loads validation images and applies rescaling
val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    'lab4/train',  # Path to the training dataset
    target_size=(img_size, img_size),  # Resize images to the specified size
    batch_size=batch_size,  # Number of images in each batch
    class_mode='binary',  # Binary classification (cats vs dogs)
    subset='validation'  # Use 20% of the training data as validation set
)

# Test Generator: Loads test images and applies rescaling
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'lab4/test',  # Path to the test dataset
    target_size=(img_size, img_size),  # Resize images to the specified size
    batch_size=batch_size,  # Number of images in each batch
    class_mode='binary'  # Binary classification (cats vs dogs)
)

# Function to plot training and validation accuracy over epochs
def plot_history(history, title):
    """
    Plots the training and validation accuracy over epochs.

    Args:
        history: History object returned by model.fit()
        title: Title of the plot
    """
    plt.plot(history.history['accuracy'], label='Training Accuracy')  # Plot training accuracy
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Plot validation accuracy
    plt.title(title)  # Set the plot title
    plt.xlabel('Epochs')  # Label x-axis
    plt.ylabel('Accuracy')  # Label y-axis
    plt.legend()  # Add legend to the plot
    plt.show()  # Display the plot

# Fully Connected Neural Network (Dense layers only)
model_fc = Sequential([
    Flatten(input_shape=(img_size, img_size, 3)),  # Flatten the 3D image input into a 1D vector
    Dense(64, activation='relu'),  # First dense layer with ReLU activation
    Dense(32, activation='relu'),  # Second dense layer with ReLU activation
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the fully connected model
model_fc.compile(optimizer='adam',  # Adam optimizer for adaptive learning rate
                 loss='binary_crossentropy',  # Loss function for binary classification
                 metrics=['accuracy'])  # Metric to track during training

# Train the fully connected model
print("\nTraining Fully Connected Network...")
history_fc = model_fc.fit(train_gen, validation_data=val_gen, epochs=initial_epochs)  # Train the model
plot_history(history_fc, "Fully Connected Network")  # Plot training and validation accuracy

# Convolutional Neural Network (CNN)
model_cnn = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),  # First convolutional layer
    MaxPooling2D((2, 2)),  # First max pooling layer to reduce spatial dimensions
    Conv2D(32, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D((2, 2)),  # Second max pooling layer to further reduce dimensions
    Flatten(),  # Flatten the 3D tensor into a 1D vector
    Dense(64, activation='relu'),  # Dense layer with ReLU activation
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the convolutional model
model_cnn.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train the convolutional model
print("\nTraining Convolutional Neural Network...")
history_cnn = model_cnn.fit(train_gen, validation_data=val_gen, epochs=initial_epochs)  # Train the model
plot_history(history_cnn, "Convolutional Neural Network")  # Plot training and validation accuracy

# Transfer Learning with VGG19
base_model_vgg = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))  # Load VGG19 model
for layer in base_model_vgg.layers:
    layer.trainable = False  # Freeze all layers to retain pre-trained weights

# Add custom classification head to VGG19
model_vgg = Sequential([
    base_model_vgg,
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the VGG19 model
model_vgg.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train the VGG19 model
print("\nTraining Transfer Learning Model (VGG19)...")
history_vgg = model_vgg.fit(train_gen, validation_data=val_gen, epochs=initial_epochs)
plot_history(history_vgg, "Transfer Learning with VGG19")

# Transfer Learning with ResNet50
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))  # Load ResNet50
for layer in base_model_resnet.layers:
    layer.trainable = False  # Freeze all layers to retain pre-trained weights

# Add custom classification head to ResNet50
model_resnet = Sequential([
    base_model_resnet,
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the ResNet50 model
model_resnet.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

# Train the ResNet50 model
print("\nTraining Transfer Learning Model (ResNet50)...")
history_resnet = model_resnet.fit(train_gen, validation_data=val_gen, epochs=initial_epochs)
plot_history(history_resnet, "Transfer Learning with ResNet50")

# Evaluate all models on the test dataset and compare performance
print("\nEvaluating Models on Test Data...")
test_accuracies = {
    "Fully Connected Network": model_fc.evaluate(test_gen)[1],
    "Convolutional Neural Network": model_cnn.evaluate(test_gen)[1],
    "Transfer Learning VGG19": model_vgg.evaluate(test_gen)[1],
    "Transfer Learning ResNet50": model_resnet.evaluate(test_gen)[1],
}

# Display test accuracies for all models
print("\nModel Performance Comparison:")
for model_name, accuracy in test_accuracies.items():
    print(f"{model_name}: {accuracy:.2f}")

# Optional: Demonstrate overfitting if selected by the user
if input("\nWould you like to demonstrate overfitting? (yes/no): ").strip().lower() == "yes":
    print("\nDemonstrating Overfitting with Increased Epochs...")
    # Train Fully Connected Network for more epochs
    history_fc_overfit = model_fc.fit(train_gen, validation_data=val_gen, epochs=overfit_epochs)
    plot_history(history_fc_overfit, "Fully Connected Network (Increased Epochs)")

    # Train Convolutional Neural Network for more epochs
    history_cnn_overfit = model_cnn.fit(train_gen, validation_data=val_gen, epochs=overfit_epochs)
    plot_history(history_cnn_overfit, "Convolutional Neural Network (Increased Epochs)")

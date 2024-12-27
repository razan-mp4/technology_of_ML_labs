import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ------------------------------
# PARAMETERS FOR TRAINING MODELS
# ------------------------------
img_size = 128  # Image dimensions (height and width) to which all images are resized
batch_size = 32  # Number of images processed in each training batch
initial_epochs = 5  # Number of training epochs for initial models
overfit_epochs = 15  # Number of epochs to demonstrate overfitting (for FCN and CNN)

# ------------------------------
# DATA PREPROCESSING USING DATA GENERATORS
# ------------------------------
# The ImageDataGenerator class is used to load images from directories and apply preprocessing
# Rescaling pixel values to the range [0, 1] using rescale=1./255

# Train Data Generator (80% of data)
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    'lab4/train',  # Path to the training dataset folder
    target_size=(img_size, img_size),  # Resize all images to (img_size, img_size)
    batch_size=batch_size,  # Batch size for training
    class_mode='binary',  # Binary classification (cats vs dogs)
    subset='training'  # Subset for training data (80%)
)

# Validation Data Generator (20% of data)
val_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    'lab4/train',  # Path to the training dataset folder
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Subset for validation data (20%)
)

# Test Data Generator (Separate dataset for testing)
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'lab4/test',  # Path to the test dataset folder
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

# ------------------------------
# FUNCTION TO PLOT ACCURACY HISTORY
# ------------------------------
def plot_history(history, title):
    """
    Plots training and validation accuracy over epochs.

    Args:
        history: The history object returned by model.fit(), which contains accuracy values.
        title: The title of the plot, specific to the model being trained.
    """
    plt.plot(history.history['accuracy'], label='Training Accuracy')  # Training accuracy plot
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Validation accuracy plot
    plt.title(title)  # Set the plot title
    plt.xlabel('Epochs')  # Label for x-axis
    plt.ylabel('Accuracy')  # Label for y-axis
    plt.legend()  # Add a legend for clarity
    plt.show()  # Display the plot

# ------------------------------
# FULLY CONNECTED NEURAL NETWORK (FCN)
# ------------------------------
# A fully connected network processes flattened (1D) data.
model_fc = Sequential([
    Flatten(input_shape=(img_size, img_size, 3)),  # Flatten 3D image input to a 1D vector
    Dense(64, activation='relu'),  # First dense hidden layer with ReLU activation
    Dense(32, activation='relu'),  # Second dense hidden layer with ReLU activation
    Dense(1, activation='sigmoid')  # Output layer for binary classification (sigmoid activation)
])

# Compile the FCN
model_fc.compile(optimizer='adam',  # Adam optimizer adjusts learning rate adaptively
                loss='binary_crossentropy',  # Binary cross-entropy loss for binary classification
                metrics=['accuracy'])  # Track accuracy during training

# Train the FCN
print("\nTraining Fully Connected Network...")
history_fc = model_fc.fit(train_gen, validation_data=val_gen, epochs=initial_epochs)
plot_history(history_fc, "Fully Connected Network")

# ------------------------------
# CONVOLUTIONAL NEURAL NETWORK (CNN)
# ------------------------------
# A CNN uses convolutional layers to extract features from images.
model_cnn = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),  # First convolutional layer
    MaxPooling2D((2, 2)),  # First max pooling layer to downsample spatial dimensions
    Conv2D(32, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D((2, 2)),  # Second max pooling layer
    Flatten(),  # Flatten the 3D output to a 1D vector
    Dense(64, activation='relu'),  # Dense layer for classification
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the CNN
model_cnn.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

# Train the CNN
print("\nTraining Convolutional Neural Network...")
history_cnn = model_cnn.fit(train_gen, validation_data=val_gen, epochs=initial_epochs)
plot_history(history_cnn, "Convolutional Neural Network")

# ------------------------------
# TRANSFER LEARNING WITH VGG19
# ------------------------------
# Load pre-trained VGG19 model without top classification layers
base_model_vgg = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
for layer in base_model_vgg.layers:
    layer.trainable = False  # Freeze all VGG19 layers to retain pre-trained features

# Add custom classification layers
model_vgg = Sequential([
    base_model_vgg,  # Pre-trained VGG19 base model
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

# ------------------------------
# TRANSFER LEARNING WITH RESNET50
# ------------------------------
# Load pre-trained ResNet50 model without top classification layers
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
for layer in base_model_resnet.layers:
    layer.trainable = False  # Freeze all ResNet50 layers

# Add custom classification layers
model_resnet = Sequential([
    base_model_resnet,  # Pre-trained ResNet50 base model
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

# ------------------------------
# EVALUATE MODELS AND COMPARE PERFORMANCE
# ------------------------------
print("\nEvaluating Models on Test Data...")
test_accuracies = {
    "Fully Connected Network": model_fc.evaluate(test_gen)[1],
    "Convolutional Neural Network": model_cnn.evaluate(test_gen)[1],
    "Transfer Learning VGG19": model_vgg.evaluate(test_gen)[1],
    "Transfer Learning ResNet50": model_resnet.evaluate(test_gen)[1],
}

# Display test accuracies
print("\nModel Performance Comparison:")
for model_name, accuracy in test_accuracies.items():
    print(f"{model_name}: {accuracy:.2f}")

# ------------------------------
# DEMONSTRATE OVERFITTING WITH ADDITIONAL EPOCHS
# ------------------------------
print("\nDemonstrating Overfitting with Increased Epochs...")

# Fully Connected Network - Additional Epochs
history_fc_overfit = model_fc.fit(train_gen, validation_data=val_gen, epochs=overfit_epochs)
plot_history(history_fc_overfit, "Fully Connected Network (Increased Epochs)")

# Convolutional Neural Network - Additional Epochs
history_cnn_overfit = model_cnn.fit(train_gen, validation_data=val_gen, epochs=overfit_epochs)
plot_history(history_cnn_overfit, "Convolutional Neural Network (Increased Epochs)")

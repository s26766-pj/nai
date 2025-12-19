import argparse
import json
import os

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np


def create_cnn_model(normalization_layer: keras.layers.Layer) -> keras.Model:
    """
    Create a simple convolutional model for Fashion MNIST.

    The model first normalizes the (28, 28) input, reshapes it to
    (28, 28, 1), applies a Conv2D + MaxPooling2D stack, then flattens
    and feeds into a dense hidden layer and a 10-way softmax output.
    """
    model = keras.Sequential(
        [
            normalization_layer,
            keras.layers.Reshape((28, 28, 1)),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def main() -> None:
    """
    Train a single CNN model with data augmentation on Fashion MNIST.

    This script:
    - Loads the Fashion MNIST dataset.
    - Normalizes the inputs with a `Normalization` layer.
    - Builds a CNN model.
    - Applies simple data augmentation using `tf.image`.
    - Trains the model and evaluates it on the test set.
    - Saves the trained model to `models/model_cnn_augment.keras`.
    - Saves metrics and plots (history, confusion matrix) to `metrics/`.
    """
    parser = argparse.ArgumentParser(
        description="Train a CNN model with augmentation on Fashion MNIST."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    args = parser.parse_args()

    # Load Fashion MNIST dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print(f"Train images dimensions: {train_images.shape}")
    print(f"Test images dimensions: {test_images.shape}")

    # Normalization layer to standardize input images; we adapt once on the training set
    normalization_layer = keras.layers.Normalization(input_shape=(28, 28), axis=None)
    normalization_layer.adapt(train_images)

    # Create CNN model using the shared normalization layer
    model = create_cnn_model(normalization_layer)

    # Ensure output directories exist for models and metrics
    models_dir = "models"
    metrics_dir = "metrics"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    base_name = "model_cnn_augment"
    model_filename = os.path.join(models_dir, f"{base_name}.keras")

    # Visualize model architecture
    keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

    # Compile the model with standard settings for classification
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Apply simple data augmentation to training images using tf.image,
    # inspired by TensorFlow's data augmentation tutorial:
    # https://www.tensorflow.org/tutorials/images/data_augmentation
    images_tf = tf.convert_to_tensor(train_images, dtype=tf.float32)
    images_tf = tf.expand_dims(images_tf, axis=-1)  # (N, 28, 28, 1)
    images_tf = tf.image.random_flip_left_right(images_tf)
    images_tf = tf.image.random_brightness(images_tf, max_delta=0.2)
    images_tf = tf.squeeze(images_tf, axis=-1)  # back to (N, 28, 28)
    train_images_aug = images_tf.numpy()

    # Train the CNN model on augmented data
    history = model.fit(train_images_aug, train_labels, epochs=args.epochs)

    # Evaluate on the original test set
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Model Accuracy: {test_acc * 100:.2f}%")

    # Predictions and confusion (error) matrix
    predictions = model.predict(test_images)
    y_pred = np.argmax(predictions, axis=1)
    confusion_matrix = tf.math.confusion_matrix(
        test_labels, y_pred, num_classes=10
    ).numpy()

    # Plot and save confusion matrix as PNG (in metrics folder)
    cm_png = os.path.join(metrics_dir, f"{base_name}_confusion.png")
    plt.figure(figsize=(6, 5))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(cm_png)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_png}")

    # Plot and save training loss/accuracy as PNG (in metrics folder)
    history_png = os.path.join(metrics_dir, f"{base_name}_history.png")
    plt.figure(figsize=(6, 4))
    plt.plot(history.history.get("loss", []), label="loss")
    if "accuracy" in history.history:
        plt.plot(history.history["accuracy"], label="accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(history_png)
    plt.close()
    print(f"Saved training history plot to {history_png}")

    # Save metrics (including loss and confusion matrix) in metrics folder
    metrics = {
        "architecture": "cnn",
        "augmentation": True,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "confusion_matrix": confusion_matrix.tolist(),
    }
    metrics_filename = os.path.join(metrics_dir, f"{base_name}_metrics.json")
    with open(metrics_filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_filename}")

    # Save the trained CNN model
    model.save(model_filename)
    print(f"Saved model to {model_filename}")


if __name__ == "__main__":
    main()


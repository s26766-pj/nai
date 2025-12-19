import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def create_cnn_model(normalization_layer: keras.layers.Layer) -> keras.Model:
    """
    Create a convolutional model for CIFAR-10 object recognition.

    The model:
    - Normalizes 32x32x3 RGB inputs using the provided normalization layer.
    - Applies two Conv2D + MaxPooling2D blocks.
    - Flattens and passes through a dense layer and a 10-way softmax output.
    """
    model = keras.Sequential(
        [
            normalization_layer,
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def main() -> None:
    """
    Train a CNN-with-augmentation model for CIFAR-10 object recognition.

    This script:
    - Loads the CIFAR-10 dataset (32x32 color images, 10 classes).
    - Normalizes inputs with a `Normalization` layer adapted on training data.
    - Builds a CNN model using `create_cnn_model`.
    - Applies simple data augmentation using `tf.image`.
    - Trains and evaluates the model.
    - Saves the trained model to `models/model_cifar10_cnn_augment.keras`.
    - Saves metrics and plots (history, confusion matrix) to `metrics/`.
    """
    parser = argparse.ArgumentParser(
        description="Train a CNN model with augmentation on CIFAR-10."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    args = parser.parse_args()

    # Load CIFAR-10 dataset
    # CIFAR-10: 50k train, 10k test, 32x32 RGB images in 10 object classes.
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    # Flatten labels to shape (N,) for convenience
    train_labels = train_labels.reshape(-1)
    test_labels = test_labels.reshape(-1)

    print(f"Train images dimensions: {train_images.shape}")
    print(f"Test images dimensions: {test_images.shape}")

    # Normalization layer: standardize RGB images (32x32x3) once on training set.
    normalization_layer = keras.layers.Normalization(input_shape=(32, 32, 3), axis=None)
    normalization_layer.adapt(train_images)

    # Build the CNN model
    model = create_cnn_model(normalization_layer)

    # Ensure output directories exist for models and metrics
    models_dir = "models"
    metrics_dir = "metrics"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    base_name = "model_cifar10_cnn_augment"
    model_filename = os.path.join(models_dir, f"{base_name}.keras")

    # Visualize model architecture
    keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

    # Compile the model with standard settings for multi-class classification
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Simple data augmentation for CIFAR-10 images using tf.image
    # Inspired by common augmentation strategies and TF tutorials:
    # https://www.tensorflow.org/datasets/catalog/cifar10?hl=en
    images_tf = tf.convert_to_tensor(train_images, dtype=tf.float32)
    # Random horizontal flip
    images_tf = tf.image.random_flip_left_right(images_tf)
    # Random brightness jitter
    images_tf = tf.image.random_brightness(images_tf, max_delta=0.2)
    # Optionally, small random contrast change
    images_tf = tf.image.random_contrast(images_tf, lower=0.8, upper=1.2)
    train_images_aug = images_tf.numpy()

    # Train the CNN model on augmented data
    history = model.fit(train_images_aug, train_labels, epochs=args.epochs, validation_split=0.1)

    # Evaluate on the original (non-augmented) CIFAR-10 test set
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # Predictions and confusion matrix
    predictions = model.predict(test_images)
    y_pred = np.argmax(predictions, axis=1)
    confusion_matrix = tf.math.confusion_matrix(
        test_labels, y_pred, num_classes=10
    ).numpy()

    # Plot and save confusion matrix as PNG (in metrics folder)
    cm_png = os.path.join(metrics_dir, f"{base_name}_confusion.png")
    plt.figure(figsize=(6, 5))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("CIFAR-10 Confusion matrix")
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
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(history_png)
    plt.close()
    print(f"Saved training history plot to {history_png}")

    # Save metrics (including loss and confusion matrix) in metrics folder
    metrics = {
        "dataset": "cifar10",
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



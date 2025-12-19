import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras


# CIFAR-10 class names in order of numeric labels 0–9.
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_and_preprocess_image(path: str) -> np.ndarray:
    """
    Load an image from disk and preprocess it for the CIFAR-10 CNN model.

    Steps:
    - Resize the image to 32x32 pixels.
    - Keep it as RGB (3 channels).
    - Return an array of shape (1, 32, 32, 3) suitable for model.predict().
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    # Load image as RGB and resize to 32x32
    img = keras.utils.load_img(path, color_mode="rgb", target_size=(32, 32))
    img_array = keras.utils.img_to_array(img)  # shape (32, 32, 3), float32 in [0, 255]

    # Add batch dimension -> (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def classify_image(model_path: str, image_path: str) -> None:
    """
    Load the trained CIFAR-10 CNN model from disk and classify a single image.

    The function:
    - Loads the Keras model from `model_path`.
    - Preprocesses the input image.
    - Runs inference and computes class probabilities.
    - Prints the predicted class name and prediction confidence.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = keras.models.load_model(model_path)

    img_batch = load_and_preprocess_image(image_path)

    # Run prediction
    preds = model.predict(img_batch)
    probs = tf.nn.softmax(preds[0]).numpy()

    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)

    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Predicted class: {class_idx} ({class_name})")
    print(f"Confidence: {confidence:.4f}")


def main() -> None:
    """
    Entry point for the CIFAR-10 prediction script in nai/lab5/task2.

    Reads the image path and optional model path from the command line,
    then performs classification and prints the result to stdout.
    """
    parser = argparse.ArgumentParser(
        description="Classify an image using the CIFAR-10 CNN model."
    )
    parser.add_argument("image", help="Path to the input image file.")
    parser.add_argument(
        "--model",
        default=os.path.join("models", "model_cifar10_cnn_augment.keras"),
        help=(
            "Path to the trained Keras model file "
            "(default: models/model_cifar10_cnn_augment.keras)."
        ),
    )
    args = parser.parse_args()

    classify_image(args.model, args.image)


if __name__ == "__main__":
    main()



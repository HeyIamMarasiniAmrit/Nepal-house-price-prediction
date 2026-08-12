from transformers import pipeline
from PIL import Image
import os

# Load the BEiT image classification model (1k ImageNet labels)
classifier = pipeline(
    "image-classification",
    model="microsoft/beit-base-patch16-224"
)

image_path = "/content/dog.png"  # change this path if needed

if os.path.exists(image_path):
    # Option 1: pass the file path directly
    predictions = classifier(image_path)

    # Option 2: load with PIL first (useful if you want to preprocess)
    # image = Image.open(image_path)
    # predictions = classifier(image)

    print("Top predictions:")
    for pred in predictions:
        print(f"{pred['label']}: {pred['score']:.4f}")
else:
    print(f"Error: File not found at {image_path}")
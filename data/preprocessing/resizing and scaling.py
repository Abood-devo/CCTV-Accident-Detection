# This code is for resizing and scaling.

import os
from PIL import Image
import shutil


base_dir = "path/to/your/label/directory"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")


TARGET_SIZE = (640, 640)


def validate_and_organize(data_dir):

    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")

    print(f"Validating and organizing directory: {data_dir}...")

    # Move images and labels into respective directories
    for file in os.listdir(data_dir):
        if file.endswith(('.jpg', '.png')):
            shutil.move(os.path.join(data_dir, file), os.path.join(image_dir, file))
        elif file.endswith('.txt'):
            shutil.move(os.path.join(data_dir, file), os.path.join(label_dir, file))

    # Validate image-label correspondence
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    labels = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    missing_labels = []
    for img in images:
        label_file = os.path.splitext(img)[0] + ".txt"
        if label_file not in labels:
            missing_labels.append(img)

    if missing_labels:
        print(f"Warning: Missing labels for {len(missing_labels)} images.")
        print("Missing:", missing_labels)
    else:
        print(f"All images in {data_dir} have corresponding labels.")

    return image_dir, label_dir


def resize_images(image_dir, target_size):

    print(f"Resizing images in {image_dir} to {target_size}...")
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png')):
            img_path = os.path.join(image_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    resized_img = img.resize(target_size)
                    resized_img.save(img_path)
            except Exception as e:
                print(f"Error resizing {img_file}: {e}")
    print("Image resizing completed.")


def normalize_labels(label_dir, image_dir, target_size):

    print(f"Normalizing labels in {label_dir}...")
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_dir, label_file)
            img_file = label_file.replace('.txt', '.jpg')
            img_path = os.path.join(image_dir, img_file)

            if not os.path.exists(img_path):
                print(f"Warning: No image found for {label_file}, skipping.")
                continue

            # Get original image dimensions
            with Image.open(img_path) as img:
                width, height = img.size

            # Read and normalize label data
            with open(label_path, 'r') as f:
                lines = f.readlines()

            normalized_lines = []
            for line in lines:
                class_id, x, y, w, h = map(float, line.strip().split())
                # Rescale relative to target dimensions
                x = x * width / TARGET_SIZE[0]
                y = y * height / TARGET_SIZE[1]
                w = w * width / TARGET_SIZE[0]
                h = h * height / TARGET_SIZE[1]
                normalized_lines.append(f"{int(class_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            # Overwrite label file
            with open(label_path, 'w') as f:
                f.writelines(normalized_lines)

    print("Label normalization completed.")


print("Starting data preparation...")
train_image_dir, train_label_dir = validate_and_organize(train_dir)
valid_image_dir, valid_label_dir = validate_and_organize(valid_dir)
resize_images(train_image_dir, TARGET_SIZE)
resize_images(valid_image_dir, TARGET_SIZE)
normalize_labels(train_label_dir, train_image_dir, TARGET_SIZE)
normalize_labels(valid_label_dir, valid_image_dir, TARGET_SIZE)
print("Data preparation completed.")
# Augmentation is applied to non-accident images, and empty .txt files are generated for them
# as they do not contain any objects (they are background images with no labels).

import os
import cv2
import albumentations as A

# Define paths
input_dir = "path/to/your/label/directory"
output_dir = "path/to/your/label/directory"
os.makedirs(output_dir, exist_ok=True)

# Define augmentation pipeline (it is based on probability)
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), p=0.5),
    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, min_holes=1, min_height=5, min_width=5, p=0.3),
])


# Function to perform augmentation and create empty txt labels
def augment_images(input_dir, output_dir, augment_count=3):
    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('jpg', 'jpeg', 'png')):  # Check for image extensions
            img_path = os.path.join(input_dir, img_name)
            image = cv2.imread(img_path)

            for i in range(augment_count):
                # Apply augmentation
                augmented = augmentation_pipeline(image=image)
                aug_img = augmented["image"]

                # Save the augmented image
                output_img_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg")
                cv2.imwrite(output_img_path, aug_img)

                # Create an empty label file with the same name as the augmented image
                label_name = os.path.splitext(img_name)[0] + f"_aug_{i}.txt"
                label_path = os.path.join(output_dir, label_name)

                # Create an empty txt file (for no accidents)
                with open(label_path, 'w') as label_file:
                    label_file.write('')  # Empty file to indicate no accident



augment_images(input_dir, output_dir)
print("Augmentation complete with empty labels!")
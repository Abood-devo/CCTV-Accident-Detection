# This code removed the images not needed in the train [0,4,9] classes in mixed or pure images.

import os
import time


def remove_images_with_classes(image_dir, label_dir, unwanted_classes=[0, 4, 9]):

    # Loop through each label file in the labels directory
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)

            with open(label_path, 'r') as file:
                lines = file.readlines()
                # Check if any unwanted class is present in the label file
                for line in lines:
                    class_id = int(line.split()[0])  # Get the class id
                    if class_id in unwanted_classes:
                        # If an unwanted class is found, delete the image and label file
                        image_name = os.path.splitext(label_file)[0] + ".jpg"  # Adjust extension if needed
                        image_path = os.path.join(image_dir, image_name)

                        # Remove image and label files with a small delay
                        try:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                print(f"Deleted image: {image_path}")
                            os.remove(label_path)
                            print(f"Deleted label: {label_path}")
                        except PermissionError:
                            print(f"PermissionError: Could not delete {label_path}, retrying...")
                            time.sleep(0.1)
                            try:
                                os.remove(label_path)
                                print(f"Deleted label (retry): {label_path}")
                            except PermissionError:
                                print(f"Still unable to delete {label_path}")
                                continue  # Skip and continue to the next file


if __name__ == "__main__":
    # Set paths to the image and label directories
    image_dir = "path/to/your/images/directory"
    label_dir = "path/to/your/labels/directory"

    # Call the function to remove images and labels with unwanted classes
    remove_images_with_classes(image_dir, label_dir)
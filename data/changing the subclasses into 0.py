# This code changes all accident-related subclass labels to class 0 (Accident),


import os

def relabel_classes_to_zero(label_dir):
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)

            # Read and process each line of the file
            with open(label_path, 'r') as file:
                lines = file.readlines()

            # Rewrite the file with updated class IDs
            with open(label_path, 'w') as file:
                for line in lines:
                    parts = line.split()  # Split YOLO line format into components
                    if len(parts) >= 5:  # Check if it's a valid YOLO line
                        parts[0] = '0'  # Set class ID to 0
                        file.write(' '.join(parts) + '\n')  # Write back as YOLO format

            print(f"Updated file: {label_path}")


if __name__ == "__main__":

    label_dir = "path/to/your/label/directory"
    relabel_classes_to_zero(label_dir)
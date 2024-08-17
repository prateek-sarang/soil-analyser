import os
import random
import shutil

# Set the path to your resized dataset
input_folder = 'C:\\Users\\Sarang Pratham\\Desktop\\opencv\\resized'

# Set the paths for the training and testing folders
train_folder = 'C:\\Users\\Sarang Pratham\\Desktop\\opencv\\Training'
test_folder = 'C:\\Users\\Sarang Pratham\\Desktop\\opencv\\Testing'

# Set the ratio of images to be used for training (e.g., 0.8 for 80% training, 20% testing)
train_ratio = 0.8

# Create the training and testing folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Iterate through each class folder in the resized dataset
for class_name in os.listdir(input_folder):
    class_path = os.path.join(input_folder, class_name)

    # Create subfolders in the training and testing folders for each class
    train_class_folder = os.path.join(train_folder, class_name)
    test_class_folder = os.path.join(test_folder, class_name)
    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(test_class_folder, exist_ok=True)

    # Get the list of images for the current class
    images = os.listdir(class_path)

    # Shuffle the list of images
    random.shuffle(images)

    # Split the images into training and testing sets
    num_train = int(train_ratio * len(images))
    train_images = images[:num_train]
    test_images = images[num_train:]

    # Copy images to the respective folders
    for img in train_images:
        src_path = os.path.join(class_path, img)
        dest_path = os.path.join(train_class_folder, img)
        shutil.copy(src_path, dest_path)

    for img in test_images:
        src_path = os.path.join(class_path, img)
        dest_path = os.path.join(test_class_folder, img)
        shutil.copy(src_path, dest_path)

print("Dataset split into training and testing sets.")

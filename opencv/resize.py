import os
import cv2

input_folder = "C:\\Users\\Sarang Pratham\\Desktop\\opencv\\Soil_types\\bruh"
output_folder = "C:\\Users\\Sarang Pratham\\Desktop\\opencv"

def resize_images(input_folder, output_folder, target_size=(224, 224)):
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        output_class_path = os.path.join(output_folder, class_name)

        os.makedirs(output_class_path, exist_ok=True)

        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            output_path = os.path.join(output_class_path, filename)

            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, target_size)

            cv2.imwrite(output_path, img_resized)

resize_images(input_folder, output_folder)

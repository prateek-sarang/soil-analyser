from PIL import Image
import os

def preprocess_dataset(input_folder, output_folder, target_size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)

    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        output_class_path = os.path.join(output_folder, class_folder)
        os.makedirs(output_class_path, exist_ok=True)

        for filename in os.listdir(class_path):
            input_path = os.path.join(class_path, filename)
            output_path = os.path.join(output_class_path, filename)

            try:
                # Open image
                img = Image.open(input_path)

                # Resize image
                img = img.resize(target_size)

                # Save the preprocessed image
                img.save(output_path)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    input_folder = "Soil_types"  # Replace with the path to your dataset
    output_folder = "preprocess"  # Specify the output folder

    preprocess_dataset(input_folder, output_folder)

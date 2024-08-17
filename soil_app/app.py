import os
from flask import Flask, render_template, request
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from soil_model import SoilModel
import torchvision.transforms as transforms
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'C:\\Users\\Sarang Pratham\\Desktop\\soil_app\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = SoilModel()  # Replace 'YourModelClass' with the actual class name of your trained model
model.load_state_dict(torch.load('soil_model.pth'))
model.eval()

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_soil(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)

    # Map predicted class index to soil type (adjust as per your model)
    soil_types = ['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5']
    predicted_soil = soil_types[predicted_class.item()]

    return predicted_soil

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Perform soil prediction
            predicted_soil = predict_soil(file_path)

            # Render the result in the HTML template
            return render_template('result.html', filename=filename, predicted_soil=predicted_soil)
        file = request.files['file']
        if file:
            print("Received image:", file.filename)
            # Add your prediction logic here
            return render_template('index.html', result="Soil prediction result goes here")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

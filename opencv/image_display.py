import cv2
import matplotlib.pyplot as plt

# Provide the correct path to your image file
image_path = r'C:\Users\Sarang Pratham\Pictures\ddd.png'

# Read the image
image = cv2.imread(image_path)

# Display the image using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes
plt.show()

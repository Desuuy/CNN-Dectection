# CNN-Dectection
This is a final exam in Deep Learning Lesson
### **Note**: The data is quite large, so I cannot upload it here. If you need it, feel free to contact me, and I will send it to you.

# Kitchen Items Recognition System

- This project aims to build a system for recognizing and localizing kitchen items using deep learning techniques.
- The system uses ResNet18 and other models for classification and localization tasks.
- It supports both training and inference functionalities, and it also provides an interactive interface using Streamlit for easy visualization.

### DataSet
- All of the data, over 6000 images, were captured by the team using a phone camera from various angles and with different materials.
- The data has been carefully and thoroughly processed by the team to train the model for optimal accuracy.
#### ⇒ The goal is to experience the entire model training process, from raw data collection, data preprocessing, to training and producing results.

## I. Required Libraries

The following libraries need to be installed:
pip install -r requirements.txt

## II.Folder Structure
```
6.ChuongTrinh
├── Data
│ └── content
│ └── drive
│ └── MyDrive
│ ├── Kitchen_items
│ │ ├── Bowl # Bowl images
│ │ ├── Chopsticks # Chopsticks images
│ │ ├── Glass # Glass images
│ │ ├── Knife # Knife images
│ │ ├── Spoon # Spoon images
│ │ └── Test # Test data
│ └── BBox # Bounding box data for localization model
│ ├── img # Images with items
│ └── labels # Coordinates files in .txt format
│ ├── train_classifier.py # Classifier model training
│ └── train_localizer.py # Localization model training
│ ├── Object_localization.ipynb
│ └── Kitchen_items_classify_Resnet_Alexnet.ipynb
├── Weights # Folder for model weights
└── dashboard.py # Streamlit dashboard application
└── app.py # Streamlit app
```
## III. Model Training Methods
itchen_items_classify_Resnet_Alexnet.ipynb:
Steps:
1. Set up environment and libraries: Import necessary libraries and connect to Google Drive for data access.
2. Prepare data: Read data from folder, process images (resize, normalization), and split into train, validation, and test sets.
3. Build the model architecture: Run Resnet18 model from notebook #10.
4. Train the model: Define training and validation functions and perform the training loop for a specified number of epochs.
5. Save the model: Save the model weights after training.

Object_localization.ipynb:
Steps:
1. Prepare data: Read data from folder containing bounding boxes and images, process and split data.

2. Build the Resnet18TwoBranch model.

3. Train and evaluate: Perform training, evaluate on validation set, and visualize results (loss, accuracy, IoU). Save plots in the "plots" folder.

4. Save the model: Save the model weights after training.

## IV. Model Inference
```
import torch
import cv2
import numpy as np
from torchvision import transforms
from models import ResNet18TwoBranch

class_id, bbox, result_img = predict('Data/content/drive/MyDrive/Kitchen_items/Test/…, 'Weights/model.pth' )

### Image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img, transform(img).unsqueeze(0)

### Prediction function
def predict(image_path, model_path):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # Process image
    original_img, img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Prediction
    with torch.no_grad():
        class_output, loc_output = model(img_tensor)
    
    # Get results
    _, predicted = torch.max(class_output, 1)
    class_id = predicted.item()
    bbox = loc_output[0].cpu().numpy()
    
    # Convert coordinates
    h, w = original_img.shape[:2]
    x_min, y_min, x_max, y_max = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
    
    # Draw bounding box
    result_img = original_img.copy()
    cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    return class_id, bbox, result_img

# Example usage
class_id, bbox, result_img = predict(
    '../Data/content/drive/MyDrive/Kitchen_items/Test/bowl_test.jpg',
    '../Weights/model.pth'
)
```
# V. Streamlit Interface Usage

1. Start Streamlit with dashboard.py in Visual Studio Code:
Open the Terminal and enter:
```
cd "E:\Nhom10_PhanMemNhanDang5VatDungTrongNha\6.ChuongTrinh"
python -m streamlit run app.py
```
2. Start the app with CMD:
```
cd E:/6.ChuongTrinh
python app.py
```
3. Interface Features:Main Interface:
  - Upload Image: Click the "Upload Image" button to select an image for recognition.
  - Recognize: Click the "Recognize" button to analyze the image.
  - Show Results: The results will display:
  - The identified item type.
  - Image with the bounding box around the item.
  - Confidence of the recognition result.

import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import random
from PIL import Image
from torchsummary import summary

# Thiết lập đường dẫn
ROOT_DIR = 'E:/6.ChuongTrinh'
DATA_DIR = os.path.join(ROOT_DIR, 'Data/content/drive/MyDrive/Kitchen_items')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'Weights')

# Đảm bảo thư mục tồn tại
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tải dữ liệu
def load_data(img_size=224):
    X = []
    Z = []
    
    # Đường dẫn đến các thư mục ảnh
    bowl_dir = os.path.join(DATA_DIR, 'Bowl')
    glass_dir = os.path.join(DATA_DIR, 'Glass')
    knife_dir = os.path.join(DATA_DIR, 'Knife')
    spoon_dir = os.path.join(DATA_DIR, 'Spoon')
    chopsticks_dir = os.path.join(DATA_DIR, 'Chopsticks')
    
    # Hàm load ảnh từ thư mục
    def make_train_data(items_dir, DIR):
        print(f"Loading images from: {DIR}")
        for img in tqdm(os.listdir(DIR)):
            label = items_dir
            path = os.path.join(DIR, img)
            try:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Không thể đọc được ảnh từ đường dẫn: {path}")
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                X.append(np.array(img))
                Z.append(str(label))
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    # Load ảnh từ các thư mục
    make_train_data('Bowl', bowl_dir)
    make_train_data('Glass', glass_dir)
    make_train_data('Knife', knife_dir)
    make_train_data('Spoon', spoon_dir)
    make_train_data('Chopsticks', chopsticks_dir)
    
    print(f"Total images loaded: {len(X)}")
    return X, Z

# Load data
X, Z = load_data()

# Chuyển nhãn thành số
label_encoder = LabelEncoder()
Z_encoded = label_encoder.fit_transform(Z)
Z_tensor = torch.tensor(Z_encoded, dtype=torch.long)

# Data Augmentation
transform = {
    "train": transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(60),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), shear=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.7, scale=(0.02, 0.3), ratio=(0.3, 3.3)),
    ]),
    "val": transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Chia dữ liệu
X_train, X_temp, y_train, y_temp = train_test_split(X, Z_tensor, test_size=0.3, random_state=42, stratify=Z_tensor)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Custom Dataset
class KitchenDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Tạo datasets và dataloaders
train_dataset = KitchenDataset(X_train, y_train, transform=transform["train"])
val_dataset = KitchenDataset(X_val, y_val, transform=transform["val"])
test_dataset = KitchenDataset(X_test, y_test, transform=transform["test"])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Tải và tùy chỉnh ResNet-18
print("Initializing ResNet18 model...")
resnet18 = models.resnet18(pretrained=True)
num_features = resnet18.fc.in_features
resnet18.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 5)
)

# Đóng băng các layers
for name, param in resnet18.named_parameters():
    if "layer3" not in name and "layer4" not in name and "fc" not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

resnet18 = resnet18.to(device)
summary(resnet18, (3, 224, 224))

# Loss và Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet18.parameters()), lr=0.00001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Huấn luyện mô hình
print("Starting training...")
train_losses = []
val_losses = []
train_accs = []
val_accs = []

num_epochs = 100
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(num_epochs):
    resnet18.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet18(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    resnet18.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels_list = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet18(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        model_path = os.path.join(WEIGHTS_DIR, 'resnet18_best_model.pth')
        torch.save(resnet18.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        counter += 1
    
    if counter >= patience:
        print("Early stopping triggered")
        break

# Hiển thị biểu đồ kết quả
plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy')
plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Train and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(WEIGHTS_DIR, 'training_curves.png'))
plt.show()

# Confusion Matrix
cm = confusion_matrix(val_labels_list, val_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(WEIGHTS_DIR, 'confusion_matrix.png'))
plt.show()

# Classification Report
print("\nClassification Report:")
report = classification_report(val_labels_list, val_preds, target_names=label_encoder.classes_)
print(report)

# Đánh giá trên tập test
print("Evaluating on test set...")
resnet18.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'resnet18_best_model.pth')))
resnet18.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet18(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")

# Lưu mô hình đầy đủ
torch.save(resnet18, os.path.join(WEIGHTS_DIR, 'resnet18_full_model.pth'))
print(f"Full model saved to {os.path.join(WEIGHTS_DIR, 'resnet18_full_model.pth')}")

# Hàm tiền xử lý và dự đoán ảnh
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def predict_image(model, image_path, label_encoder):
    model.eval()
    image = preprocess_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return label_encoder.inverse_transform([predicted.item()])[0]

# Chọn ngẫu nhiên ảnh từ thư mục Test và dự đoán (nếu có)
test_dir = os.path.join(DATA_DIR, 'Test')
if os.path.exists(test_dir):
    print(f"Testing on sample images from {test_dir}...")
    all_images = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if all_images:
        selected_images = random.sample(all_images, min(25, len(all_images)))
        
        # Dự đoán và lưu kết quả
        predictions = []
        original_images = []
        for img_path in selected_images:
            pred_label = predict_image(resnet18, img_path, label_encoder)
            predictions.append(pred_label)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_images.append(img)
        
        # Hiển thị kết quả
        plt.figure(figsize=(15, 15))
        for i in range(len(selected_images)):
            plt.subplot(5, 5, i + 1)
            plt.imshow(original_images[i])
            plt.title(f"Pred: {predictions[i]}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(WEIGHTS_DIR, 'test_predictions.png'))
        plt.show()

print("Training and evaluation completed successfully!")
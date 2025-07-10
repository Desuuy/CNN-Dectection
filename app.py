import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models  # Thêm import models
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os

# Thiết lập đường dẫn
ROOT_DIR = 'E:/6.ChuongTrinh'
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'Weights')
MODEL_PATH = os.path.join(WEIGHTS_DIR, 'model.pth')

# Định nghĩa lớp ResNet18TwoBranch giống như trong mã huấn luyện
class ResNet18TwoBranch(torch.nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet18TwoBranch, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.features = torch.nn.Sequential(*list(resnet18.children())[:-2])

        self.new_conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )

        self.conv = torch.nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(256)
        self.relu = torch.nn.ReLU(inplace=True)

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()

        self.cls_branch = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.8),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.8),
            torch.nn.Linear(128, num_classes)
        )

        self.loc_branch = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.8),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.8),
            torch.nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.new_conv_block(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)

        class_output = self.cls_branch(x)
        loc_output = torch.sigmoid(self.loc_branch(x))
        return class_output, loc_output

# Tạo một container cho thông báo tải mô hình
loading_container = st.empty()
loading_container.text("Đang tải mô hình...")

try:
    # Tải mô hình từ file .pth (cả kiến trúc và trọng số)
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()  # Chuyển mô hình sang chế độ đánh giá (evaluation mode)
    loading_container.text("Mô hình đã được tải thành công!")
except Exception as e:
    loading_container.error(f"Lỗi khi tải mô hình: {str(e)}")
    st.stop()  # Dừng ứng dụng nếu không tải được mô hình

# Ánh xạ class_id sang tên lớp
class_names = {0: "Bowl", 1: "Chopsticks", 2: "Glass", 3: "Knife", 4: "Spoon", 5: "Background"}

# Tiêu đề của ứng dụng
st.title("Ứng dụng Nhận dạng Vật dụng trong Nhà")
st.write("Tải lên hình ảnh để nhận dạng các vật dụng: bát, đũa, ly, dao, muỗng")

# Tải hình ảnh từ người dùng
uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc hình ảnh
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hình ảnh đã tải lên", use_container_width=True)

    # Tiền xử lý hình ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Điều chỉnh kích thước theo yêu cầu của mô hình
        transforms.ToTensor(),  # Chuyển thành tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
    ])
    img_tensor = transform(image).unsqueeze(0)  # Thêm chiều batch: (1, 3, 224, 224)

    # Dự đoán với mô hình
    with torch.no_grad():  # Tắt tính toán gradient để tăng tốc độ
        class_output, loc_output = model(img_tensor)

    # Xử lý đầu ra
    # class_output: Xác suất cho từng lớp (kích thước: [1, num_classes])
    # loc_output: Tọa độ hộp bao đã chuẩn hóa (kích thước: [1, 4])
    class_scores = torch.softmax(class_output, dim=1).squeeze(0).numpy()  # Xác suất cho từng lớp
    class_id = np.argmax(class_scores)  # Lớp có xác suất cao nhất
    score = class_scores[class_id]  # Xác suất cao nhất

    # loc_output đã được chuẩn hóa qua sigmoid (giá trị trong [0, 1])
    boxes = loc_output.squeeze(0).numpy()  # [x_min, y_min, x_max, y_max]

    # Vẽ hộp bao và nhãn lên hình ảnh
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    if score > 0.33 and class_names[class_id] != "Background":  # Chỉ hiển thị các dự đoán có độ tin cậy cao và không phải Background
        # Chuyển tọa độ chuẩn hóa về tọa độ pixel
        x_min = boxes[0] * img_width
        y_min = boxes[1] * img_height
        x_max = boxes[2] * img_width
        y_max = boxes[3] * img_height

        # Vẽ hộp bao
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

        # Tạo nhãn với tên lớp và độ tin cậy
        label = f"{class_names[class_id]}: {score:.2f}"

        # Sử dụng font với kích thước lớn hơn
        try:
            font = ImageFont.truetype("arial.ttf", 20)  # Font Arial, kích thước điều chỉnh
        except:
            font = ImageFont.load_default()  # Sử dụng font mặc định nếu không tìm thấy Arial

        # Tính kích thước của nhãn
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Thêm padding cho nền của nhãn
        padding = 5
        bg_width = text_width + 2 * padding
        bg_height = text_height + 2 * padding

        # Vị trí nền: phía trên hộp bao
        bg_x = x_min
        bg_y = y_min - bg_height - 5 if y_min - bg_height - 5 > 0 else y_min + 5

        # Vẽ nền cho nhãn
        draw.rectangle(
            [bg_x, bg_y, bg_x + bg_width, bg_y + bg_height],
            fill=(255, 255, 255, 200)  # Màu trắng với độ trong suốt
        )

        # Căn giữa nhãn trong nền
        text_x = bg_x + padding
        text_y = bg_y + padding

        # Vẽ nhãn lên hình ảnh
        draw.text((text_x, text_y), label, fill="red", font=font)

        # Hiển thị thông tin chi tiết
        st.success(f"Đã phát hiện: {class_names[class_id]} với độ tin cậy {score:.2f}")
        st.info(f"Tọa độ hộp giới hạn (chuẩn hóa): [{boxes[0]:.2f}, {boxes[1]:.2f}, {boxes[2]:.2f}, {boxes[3]:.2f}]")
    else:
        if class_names[class_id] == "Background":
            st.info("Không phát hiện vật dụng nào trong ảnh.")
        else:
            st.warning(f"Phát hiện không đáng tin cậy: {class_names[class_id]} (độ tin cậy: {score:.2f})")

    # Hiển thị hình ảnh với hộp bao và nhãn
    st.image(image, caption="Hình ảnh với các đối tượng được phát hiện", use_container_width=True)
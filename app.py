from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

# Load trained model
import torchvision.models as models

# Import DenseNet
model = torch.load('D:\\Năm 4\\KLTN\\OCT-classification-tool\\static\\models\\densenet201-26.h5', map_location=torch.device('cpu'))
model.eval()

mean = np.array([0.19092108, 0.19092108, 0.19092108])
std = np.array([0.20110247, 0.20110247, 0.20110247])
# List of class labels
labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def model_predict(img_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    preds = model(input_batch)
    # Convert the prediction output to probabilities
    probabilities = torch.nn.functional.softmax(preds, dim=1)

    # Get the index of the predicted class label
    _, predicted_idx = torch.max(probabilities, 1)
    predicted_label = predicted_idx.item()
    return preds, probabilities[0][predicted_label].item(), predicted_label

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Lấy file hình ảnh từ yêu cầu POST
        file = request.files['image']
        # Lưu file vào thư mục UPLOAD_FOLDER
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Dự đoán với mô hình đã tải
        preds, probabilities, predicted_label = model_predict(file_path, model)
        # Xử lý kết quả dự đoán
        # ...
        # Chuẩn bị kết quả dự đoán để truyền vào template
        predicted_class = 'Predicted class: ' + str(preds)
        # Trả về template với kết quả dự đoán
        return render_template('result.html', predicted_class=predicted_class, probabilities=probabilities, predicted_label=labels[predicted_label])

if __name__ == '__main__':
    app.run()
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import *
import math
from sklearn.preprocessing import label_binarize

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

model_dirs = [
    'D:\\Năm 4\\KLTN\\OCT-classification-tool\\static\\models\\densenet201-26.h5',
    'D:\\Năm 4\\KLTN\\OCT-classification-tool\\static\\models\\efficientnet-b3-29.h5',
    'D:\\Năm 4\\KLTN\\OCT-classification-tool\\static\\models\\inception_v3-16.h5',
    'D:\\Năm 4\\KLTN\\OCT-classification-tool\\static\\models\\resnet50-5.h5'
]
model_names = [
    'DenseNet201', 'EfficientNet-B3', 'Inception-V3', 'ResNet50'
]

mean = np.array([0.19092108, 0.19092108, 0.19092108])
std = np.array([0.20110247, 0.20110247, 0.20110247])

# List of class labels
labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def model_predict(img_path, model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    outputs = model(input_batch)
    prob = torch.nn.functional.softmax(outputs, dim=1)
    
    return prob

ensemble_functions = ['DefaultGompertz', 'Gompertz', 'ShiftGompertz', 'WeightedGompertz', 'Exponential', 'DefaultExponential', 'NewBurr', 'Gamma', 'ExponentialGompertz', 'ExponentialTangent', 'Average']

def fuzzy_rank(CF, top):
    R_L = np.zeros(CF.shape)
    for i in range(CF.shape[0]):
        for j in range(CF.shape[1]):
            for k in range(CF.shape[2]):
                R_L[i][j][k] = 1 - math.exp(-math.exp(-2.0*CF[i][j][k]))  #Gompertz Function
    
    K_L = 0.632*np.ones(shape = R_L.shape) #initiate all values as penalty values
    for i in range(R_L.shape[0]):
        for sample in range(R_L.shape[1]):
            for k in range(top):
                a = R_L[i][sample]
                idx = np.where(a==np.partition(a, k)[k])
                #if sample belongs to top 'k' classes, R_L =R_L, else R_L = penalty value
                K_L[i][sample][idx] = R_L[i][sample][idx]

    return K_L

def CFS_func(CF, K_L):
    H = CF.shape[0] #no. of classifiers
    for f in range(CF.shape[0]):
        for i in range(CF.shape[1]):
            idx = np.where(K_L[f][i] == 0.632)
            CF[f][i][idx] = 0
    CFS = 1 - np.sum(CF,axis=0)/H
    return CFS

def Gompertz(top = 2, *argv):
    L = 0 #Number of classifiers
    for arg in argv:
        L += 1

    num_classes = arg.shape[1]
    CF = np.zeros(shape = (L,arg.shape[0], arg.shape[1]))

    for i, arg in enumerate(argv):
        CF[:][:][i] = arg

    R_L = fuzzy_rank(CF, top) #R_L is with penalties
    
    RS = np.sum(R_L, axis=0)
    CFS = CFS_func(CF, R_L)
    FS = RS*CFS

    predictions = np.argmin(FS,axis=1)
    return predictions

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
        outputs = [] 
        predicts = []
        # Dự đoán với mô hình đã tải
        for model in model_dirs:
            output = model_predict(file_path, model)  # Lưu kết quả dự đoán vào biến result
            _, predict = torch.max(output, 1)
            label = labels[predict.item()]
            confidence = output[0][predict.item()].item()
            predicts.append([label, confidence])
            outputs.append(np.asarray(output[0].tolist()[0:4]))
        
        p1 = np.array([outputs[0]])
        p2 = np.array([outputs[1]])
        p3 = np.array([outputs[2]])
        p4 = np.array([outputs[3]])
        top = 4 #top 'k' classes
        predictions = Gompertz(top, p1, p2, p3, p4)
        label = labels[predictions[0]]
        return render_template('result.html', labels=labels, model_names=model_names, outputs=outputs, predicts=predicts, predictions=label)

if __name__ == '__main__':
    app.run()
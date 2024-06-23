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
    ['DenseNet201', 'static/models/densenet201-26.h5'],
    ['EfficientNet-B3', 'static/models/efficientnet-b3-29.h5'],
    ['Inception-V3', 'static/models/inception_v3-16.h5'],
    ['ResNet50', 'static/models/resnet50-5.h5']
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


# 1 Gompertz (Go) 
def DefaultGompertz(x):
    return 1 - math.exp(-math.exp(-2.0* x))  #Default Gompertz Function

# 3 Exponential (Ex) 
def DefaultExponential(x): 
    return 1 - math.exp(-((x-1)**2)/2.0) # Exponential Function

# 2 Exponential * Tangent (ExTan) 
def DefaultTangent(x):
    return 1 - math.tanh(((x-1)**2)/2) # DefaultTangent Function

def ExponentialTangent(x):
    # return (1 - math.exp(-((x-1)**2)/2.0)) * (1 - math.tanh(((x-1)**2)/2)) # exponential tangent
    return DefaultExponential(x) * DefaultTangent(x) # exponential tangent

# 4 Estimated Gompertz (EGo)
def Gompertz(x):
    b = 0.473
    eta = 1.846
    #b = 17.5
    #eta = 0.000001
    #b = 3.07146
    return 1 - math.exp(-eta * (math.exp(-b *x)-1))  #Re-paremeter Gompertz Function

# 5 Estimated Weighted Gompertz (EWGo)
def WeightedGompertz(x):
    s = 1.97 #1.87
    l = 0.82275#82275 #
    #l = 1.8324 #bỏ -1
    #s= 0.5637 #bỏ -1
    #s= 1.8755 #
    #l = 1.837#1.8373 #
    return 1 - (1+(s * (math.exp(l*x)-1))/(1+l*s))* math.exp(s * (math.exp(l*x)-1)) # Weighted Gompertz
    #return 1 - (1+(s * (math.exp(l*x)))/(1+l*s))* math.exp(-s * (math.exp(l*x))) # Weighted Gompertz
    
# 6 Estimated New Burr (ENB)
def NewBurr(x):
    p = 0.0000006156
    #p = 0.0000009285
    return (1+ math.exp(- x**3))**(-p) # New Burr Function

# Get penalty với x = 0
def get_penalty(function):
    if function =='DefaultGompertz':
        return DefaultGompertz(0)
    if function =='ExponentialTangent':
        return ExponentialTangent(0)
    if function =='DefaultExponential':
        return DefaultExponential(0)
    if function =='Gompertz':
        return Gompertz(0) 
    if function =='WeightedGompertz':
        return WeightedGompertz(0)
    if function =='NewBurr':
        return NewBurr(0)
    
ensemble_functions = [['DefaultGompertz', 'Gompertz (Go)'], 
                      ['ExponentialTangent', 'Exponential * Tangent (ExTan)'], 
                      ['DefaultExponential', 'Exponential (Ex)'], 
                      ['Gompertz', 'Estimated Gompertz (EGo)'], 
                      ['WeightedGompertz', 'Estimated Weighted Gompertz (EWGo)'], 
                      ['NewBurr', 'Estimated New Burr (ENB)']]

def fuzzy_rank(CF, top, function, penalty):
    R_L = np.zeros(CF.shape)
    for i in range(CF.shape[0]):
        for j in range(CF.shape[1]):
            for k in range(CF.shape[2]):
                if function =='DefaultGompertz':
                    R_L[i][j][k] = DefaultGompertz(CF[i][j][k])
                if function =='ExponentialTangent':
                    R_L[i][j][k] = ExponentialTangent(CF[i][j][k])
                if function =='DefaultExponential':
                    R_L[i][j][k] = DefaultExponential(CF[i][j][k])
                if function =='Gompertz':
                    R_L[i][j][k] = Gompertz(CF[i][j][k])
                if function =='WeightedGompertz':
                    R_L[i][j][k] = WeightedGompertz(CF[i][j][k])
                if function =='NewBurr':
                    R_L[i][j][k] = NewBurr(CF[i][j][k])
                
    #default gompertz penalty 0.632
    K_L = penalty * np.ones(shape = R_L.shape) #initiate all values as penalty values
    #print ("K_L: ", K_L)
    for i in range(R_L.shape[0]):
        for sample in range(R_L.shape[1]):
            for k in range(top):
                a = R_L[i][sample]
                idx = np.where(a==np.partition(a, k)[k])
                #if sample belongs to top 'k' classes, R_L =R_L, else R_L = penalty value
                K_L[i][sample][idx] = R_L[i][sample][idx]
    # print ("K_L: ", K_L)
    return K_L

def CFS_func(CF, K_L, penalty):
    H = CF.shape[0] #no. of classifiers
    for f in range(CF.shape[0]):
        for i in range(CF.shape[1]):
            idx = np.where(K_L[f][i] == penalty) #default gompertz penalty 0.632
            CF[f][i][idx] = 0
    CFS = 1 - np.sum(CF,axis=0)/H
    return CFS

def Distributions(function, top = 2, *argv):
    L = 0 #Number of classifiers
    for arg in argv:
        L += 1
    
    num_classes = arg.shape[1]
    CF = np.zeros(shape = (L,arg.shape[0], arg.shape[1]))
    penalty = get_penalty(function)
    # print ("Penalty: ", penalty)
    for i, arg in enumerate(argv):
        CF[:][:][i] = arg
    # print (CF.shape)
    R_L = fuzzy_rank(CF, top, function, penalty) #R_L is with penalties
    RS = np.sum(R_L, axis=0)
    CFS = CFS_func(CF, R_L, penalty)
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
        filename = filename.split("-")[0] + '.' + filename.split(".")[-1]
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        outputs = []
        predicts = []
        selected_radio = request.form['selected']
        
        if selected_radio == 'model':
            selected_models = request.form.getlist('modelGroup')
            # Xử lý danh sách các model được chọn
            for model in model_dirs :
                if model[0] in selected_models :
                    output = model_predict(file_path, model[1])
                    _, predict = torch.max(output, 1)
                    label = labels[predict.item()]
                    confidence = output[0][predict.item()].item()
                    formatConfidence = "{:.2f}%".format(round(confidence * 100, 2))
                    predicts.append([model[0], label, formatConfidence])
                    outputs.append(np.asarray(output[0].tolist()[0:4]))
            
            return render_template('result.html', labels=labels, outputs=outputs, predicts=predicts, predictions=[])
    
        else :
             # Dự đoán với mô hình đã tải
            for model in model_dirs:
                output = model_predict(file_path, model[1])
                _, predict = torch.max(output, 1)
                label = labels[predict.item()]
                confidence = output[0][predict.item()].item()
                formatConfidence = "{:.2f}%".format(round(confidence * 100, 2))
                predicts.append([model[0], label, formatConfidence])
                outputs.append(np.asarray(output[0].tolist()[0:4]))
        
            p1 = np.array([outputs[0]])
            p2 = np.array([outputs[1]])
            p3 = np.array([outputs[2]])
            p4 = np.array([outputs[3]])
            top = 4 #top 'k' classes
            predictions = []
            
            if selected_radio == 'ensemble':
                selected_ensemble = request.form.getlist('ensembleGroup')
                # Xử lý danh sách các ensemble được chọn
                for ensemble in ensemble_functions : 
                    if ensemble[0] in selected_ensemble :
                        prediction = Distributions(ensemble[0], top, p1, p2, p3, p4)
                        predictions.append([ensemble[1], labels[prediction[0]]])
            else :
                for ensemble in ensemble_functions : 
                    prediction = Distributions(ensemble[0], top, p1, p2, p3, p4)
                    predictions.append([ensemble[1], labels[prediction[0]]])
        
            return render_template('result.html', labels=labels, outputs=outputs, predicts=predicts, predictions=predictions)

if __name__ == '__main__':
    app.run()
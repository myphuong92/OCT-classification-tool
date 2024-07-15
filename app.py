from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import *
import math

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model_dirs = [
    ['Inception-V3', 'static/models/inception_v3-16.h5'],
    ['ResNet50', 'static/models/resnet50-5.h5'],
    ['DenseNet201', 'static/models/densenet201-26.h5'],
    ['EfficientNet-B3', 'static/models/efficientnet-b3-29.h5'],
]

mean = np.array([0.19092108, 0.19092108, 0.19092108])
std = np.array([0.20110247, 0.20110247, 0.20110247])

# List of class labels
labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def model_predict(img_path, model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    modelName = model_path.split('/')[2]
    resolution = 224
    if modelName == 'inception_v3-16':
        resolution = 299
    if modelName == 'efficientnet-b3-29':
        resolution = 300
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
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

# 2 Exponential (Ex) 
def DefaultExponential(x): 
    return 1 - math.exp(-((x-1)**2)/2.0) # Exponential Function

# 3 Exponential * Tangent (ExTan) 
def ExponentialTangent(x):
    return (1 - math.exp(-((x-1)**2)/2.0)) * (1 - math.tanh(((x-1)**2)/2)) # exponential tangent

# 4 Exponential * TanH * Sigmoid (ExTanSig)
def ExponentialTangentSigmoid(x):
    return (1 - math.exp(-((x-1)**2)/2.0)) * (1 - math.tanh(((x-1)**2)/2)) * (1/(1 + math.exp(-x))) # exponential tangent sigmoid
    
# 5 Mitscherlich  (Mi)
def DefaultMitscherlich(x):
    return 2*(1- 2**(x-1)) # DefaultMitscherlich Function

# 6 Estimated Gompertz (EGo)
def Gompertz(x):
    #b = 0.473 có trừ 1
    eta = 1.846
    b = 17.52
    #eta = 0.000001
    return 1 - math.exp(-eta * (math.exp(-b *x)))  #Re-paremeter Gompertz Function

# 7 Modified Gamma (MGa)
def Gamma(x):
    #b = 1.96574
    #s = 1.95654
    #return 1 - math.exp(-b * s * x) # Gamma Function
    return math.gamma(x+1)

# Get penalty với x = 0
def get_penalty(function):
    if function =='DefaultGompertz':
        return DefaultGompertz(0)
    if function =='DefaultExponential':
        return DefaultExponential(0)
    if function =='ExponentialTangent':
        return ExponentialTangent(0)
    if function =='ExponentialTangentSigmoid':
        return ExponentialTangentSigmoid(0) 
    if function =='DefaultMitscherlich':
        return DefaultMitscherlich(0)
    if function =='Gompertz':
        return Gompertz(0) 
    if function =='Gamma':
        return Gamma(0) 
    
ensemble_functions = [['DefaultGompertz', 'Gompertz (Go)'], 
                      ['DefaultExponential', 'Exponential (Ex)'], 
                      ['ExponentialTangent', 'Exponential * Tangent (ExTan)'],
                      ['ExponentialTangentSigmoid', 'Exponential * TanH * Sigmoid (ExTanSig)'],
                      ['DefaultMitscherlich', 'Mitscherlich  (Mi)'],
                      ['Gompertz', 'Estimated Gompertz (EGo)'], 
                      ['Gamma', 'Modified Gamma (MGa)'],
                    ]

def fuzzy_rank(CF, top, function, penalty):
    R_L = np.zeros(CF.shape)
    for i in range(CF.shape[0]):
        for j in range(CF.shape[1]):
            for k in range(CF.shape[2]):
                if function =='DefaultGompertz':
                    R_L[i][j][k] = DefaultGompertz(CF[i][j][k])
                if function =='DefaultExponential':
                    R_L[i][j][k] = DefaultExponential(CF[i][j][k])
                if function =='ExponentialTangent':
                    R_L[i][j][k] = ExponentialTangent(CF[i][j][k])
                if function =='ExponentialTangentSigmoid':
                    R_L[i][j][k] = ExponentialTangentSigmoid(CF[i][j][k])
                if function =='DefaultMitscherlich':
                    R_L[i][j][k] = DefaultMitscherlich(CF[i][j][k])
                if function =='Gompertz':
                    R_L[i][j][k] = Gompertz(CF[i][j][k])
                if function =='Gamma':
                    R_L[i][j][k] = Gamma(CF[i][j][k])
                
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
    #CFS = 1 - np.sum(CF,axis=0)/H
    CFS = 1 - np.sum(CF,axis=0)/H
    return CFS

def Distributions(function, top = 4, argv = []):
    L = 0 #Number of classifiers
    
    for arg in argv:
        L += 1
    
    # num_classes = arg.shape[1]
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
        # data_dir = 'D:\\Quynh\\algorithm\\data\\cval'
        # results = []
        # i=0
        # for label in labels:
        #     label_dir = os.path.join(data_dir, label)
    
        #     for filename in os.listdir(label_dir):
        #         img_path = os.path.join(label_dir, filename)
        #         result = []
        #         result.append(filename)
        #         label_index = labels.index(label)
        #         result.append(label_index)
        #         outputs = []
        #         for model in model_dirs :
        #             output = model_predict(img_path, model[1])
        #             _, predict = torch.max(output, 1)
        #             label = labels[predict.item()]
        #             outputs.append(np.asarray(output[0].tolist()[0:4]))
        #             result.append(predict.item())
        #         p = []
        #         p3 = []
        #         j=0
        #         for output in outputs:
        #             p.append(np.array([output]))
        #             if j < 3 :
        #                 p3.append(np.array([output]))
        #             j=j+1
        #         top = 2 #top 'k' classes
        #         for ensemble in ensemble_functions : 
        #             prediction = Distributions(ensemble[0], top, p)
        #             result.append(prediction[0])
                    
        #             prediction3 = Distributions(ensemble[0], top, p3)
        #             result.append(prediction3[0])
        #         results.append(result)
        #         print(i)
        #         i= i + 1
        # import csv
        # # Đường dẫn tới tệp tin CSV để lưu
        # csv_file = 'results.csv'

        # # Mở tệp tin CSV để ghi
        # with open(csv_file, 'w', newline='') as file:
        #     writer = csv.writer(file)
    
        #     # Ghi tiêu đề cột
        #     writer.writerow(['Filename', 'Label', 'Inception-V3', 'ResNet50', 'DenseNet201', 'EfficientNet-B3', 
        #                      'Gompertz (Go)', 'Go3',
        #                      'Exponential (Ex)', 'Ex3',
        #                      'Exponential * Tangent (ExTan)', 'ExTan3',
        #                      'Exponential * TanH * Sigmoid (ExTanSig)', 'ExTanSig',
        #                      'Mitscherlich  (Mi)', 'Mi3',
        #                      'Estimated Gompertz (EGo)', 'EGo3',
        #                      'Modified Gamma (MGa)', 'MGa3'])
    
        #     # Ghi từng kết quả vào tệp tin CSV
        #     for result in results:
        #         writer.writerow(result)
        
        # return render_template('index.html')
        
        # Lấy file hình ảnh từ yêu cầu POST
        file = request.files['image']
        # Lưu file vào thư mục UPLOAD_FOLDER
        filename = secure_filename(file.filename)
        trueLabel = filename.split("-")[0]
        extension = filename.split(".")[-1]
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], trueLabel + '.' + extension)
        file.save(file_path)
        outputs = []
        predicts = []
        selected_radio = request.form['selected']
        
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
        
        p = []
        for output in outputs:
            p.append(np.array([output]))
        
        print(outputs)
        print("p: ")
        print(p)
        top = 4 #top 'k' classes
        predictions = []
            
        if selected_radio == 'ensemble':
            selected_ensemble = request.form.getlist('ensembleGroup')
            # Xử lý danh sách các ensemble được chọn
            for ensemble in ensemble_functions : 
                if ensemble[0] in selected_ensemble :
                    prediction = Distributions(ensemble[0], top, p)
                    predictions.append([ensemble[1], labels[prediction[0]]])
        
        return render_template('result.html', image=file_path, trueLabel=trueLabel, filename=filename, labels=labels, outputs=outputs, predicts=predicts, predictions=predictions)

if __name__ == '__main__':
    app.run()
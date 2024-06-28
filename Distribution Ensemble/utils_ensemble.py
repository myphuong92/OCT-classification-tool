import pandas as pd
import numpy as np
from sklearn.metrics import *
import math, os
from sklearn.preprocessing import label_binarize

penalty=0

def getfile(filename):
    root = ''
    test_dir = "/content/drive/My Drive/KHOA HTTT/Giảng dạy HTTT/KLTN/Hướng dẫn/2024/KLTN - Phương, Quỳnh 2020 - OCT/Dataset OCT/test/"
    
    file = root + filename
    if '.csv' not in filename:
        file = file + '.csv'
    
    df = pd.read_csv(file, header=None)
    df = np.asarray(df)

    labels = []
    file_list = sorted(os.listdir(root + test_dir))  # Sắp xếp danh sách tệp theo thứ tự tăng dần
    for i, c in enumerate(file_list):
        for j in range(len(os.listdir(root + test_dir + c))):
            labels.append(i)
    labels = np.asarray(labels)
    return df, labels

#ROC-AUC
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
def plot_roc(val_label,decision_val, caption='ROC Curve'):
    num_classes=np.unique(val_label).shape[0]
    classes = []
    for i in range(num_classes):
        classes.append(i)
    plt.figure()
    decision_val = label_binarize(decision_val, classes=classes)
    
    if num_classes!=2:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            y_val = label_binarize(val_label, classes=classes)
            fpr[i], tpr[i], _ = roc_curve(y_val[:, i], decision_val[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i+1, roc_auc[i]))
    else:
        fpr,tpr,_ = roc_curve(val_label,decision_val, pos_label=1)
        roc_auc = auc(fpr,tpr)*100
        plt.plot(fpr,tpr,label='ROC curve (AUC=%0.2f)'%roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(caption)
    plt.legend(loc="lower right")
    plt.savefig(str(len(classes))+'.png',dpi=300)

def predicting(ensemble_prob):
    prediction = np.zeros((ensemble_prob.shape[0],))
    for i in range(ensemble_prob.shape[0]):
        temp = ensemble_prob[i]
        t = np.where(temp == np.max(temp))[0][0]
        prediction[i] = t
    return prediction

def metrics(labels,predictions,classes):
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names = classes,digits = 4))
    matrix = confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(matrix)
    print("\nClasswise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
    print("\nBalanced Accuracy Score: ",balanced_accuracy_score(labels,predictions))

# Định nghĩa các hàm phân phối
def DefaultGompertz(x):
    return 1 - math.exp(-math.exp(-2.0* x))  #Default Gompertz Function

def Gompertz(x):
    #b = 0.473 có trừ 1
    eta = 1.846
    b = 17.52
    #eta = 0.000001
    return 1 - math.exp(-eta * (math.exp(-b *x)))  #Re-paremeter Gompertz Function

def ShiftGompertz(x):
    #eta = -0.0932
    #b = 2.0829
    eta = 0.831 # chuyển qua số dương, bỏ - trước eta
    b = 2.036
    #eta = 1.132
    #b = 3.423
    return (1- math.exp(-b * x)) * math.exp(-eta * math.exp(-b * x)) # Shilf Gompertz

def WeightedGompertz(x):
    s = 1.97 #1.87
    #s= 1.8324379920959473
    l = 0.82275#82275 #
    #l = 0.5636825561523438

    return 1 - (1+(s * (math.exp(l*x)-1))/(1+l*s))* math.exp(s * (math.exp(l*x)-1)) # Weighted Gompertz
    #return 1 - (math.exp(-l*x))* math.exp(s * (math.exp(-l*x))) # Weighted Gompertz

def DefaultExponential(x):
    return 1 - math.exp(-((x-1)**2)/2.0) # Exponential Function

def DefaultTangent(x):
    return 1 - math.tanh(((x-1)**2)/2) # DefaultTangent Function

def Sigmoid(x):
    return 1/(1 + math.exp(-x)) # Sigmoid Function

def DefaultMitscherlich(x):
    return 2*(1- 2**(x-1)) # DefaultMitscherlich Function

def Mitscherlich(x):
    a = 1.8855880498886108
    c = 1.908218264579773
    a= 10.9839448928833
    c = 10.998953819274902
    a= 1.8855880498886108
    c = 1.7366911172866821
    b= 1.653203010559082
    #return a*(1- math.exp(-c * x)) # SigmoMitscherlichid Function
    return  a*(1- math.exp(-c * (x+b)))

def Exponential(x):
    #l = 1.8992
    #l = 1.8827
    l = 2.425
    l = 1.9082
    return 1 - math.exp(-l * x) # Exponential Function

def Gamma(x):
    #b = 1.96574
    #s = 1.95654
    #return 1 - math.exp(-b * s * x) # Gamma Function
    return math.gamma(x+0.1)

def Ibrahim(x):
    l = 1.0853286981582642
    b = 2.9221911430358887
    return 1 - (1/(l+1))*(math.exp(-b*x)+l*math.exp(-(b*x)**2)) # Ibrahim Function

def Logistic(x):
    b0 = 10.775835990905762
    b1 = 9.2244291305542
    return 1/(1+ math.exp(-(b0+ b1 *x))) # logistic Function


def NewBurr(x):
    return (1+ math.exp(- x**3))/2 # New Burr Function

def ExponentialGompertz(x):
    return DefaultExponential(x) * Gompertz(x)

def ExponentialTangent(x):
    return (1 - math.exp(-((x-1)**2)/2.0)) * (1 - math.tanh(((x-1)**2)/2)) # exponential tangent

def ExponentialTangentSigmoid(x):
    return (1 - math.exp(-((x-1)**2)/2.0)) * (1 - math.tanh(((x-1)**2)/2)) * (1/(1 + math.exp(-x))) # exponential tangent sigmoid

def ExponentialSigmoid(x):
    return (1 - math.exp(-((x-1)**2)/2.0)) * (1/(1 + math.exp(-x))) # exponential sigmoid

def Average(x):

    arr = np.array([NewBurr(x),Gompertz(x)])
    normalized_arr = np.apply_along_axis(normalize_array, axis=0, arr=arr)

    #print (arr)
    #return np.prod(arr)
    return np.max(arr)
    #return np.mean(arr)

def normalize_array(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


# Get penalty với x = 0
def get_penalty(function):
    if function =='DefaultGompertz':
        return DefaultGompertz(0)
    if function =='Gompertz':
        return Gompertz(0)
    if function =='ShiftGompertz':
        return ShiftGompertz(0)
    if function =='WeightedGompertz':
        return WeightedGompertz(0)
    if function =='Exponential':
        return Exponential(0)
    if function =='DefaultExponential':
        return DefaultExponential(0)
    if function =='DefaultTangent':
        return DefaultTangent(0)
    if function =='Sigmoid':
        return Sigmoid(0)
    if function =='Mitscherlich':
        return Mitscherlich(0)
    if function =='DefaultMitscherlich':
        return DefaultMitscherlich(0)
    if function =='Logistic':
        return Logistic(0)
    if function =='NewBurr':
        return NewBurr(0)
    if function =='Ibrahim':
        return Ibrahim(0)
    if function =='Gamma':
        return Gamma(0)
    if function =='ExponentialGompertz':
        return ExponentialGompertz(0)
    if function =='ExponentialTangent':
        return ExponentialTangent(0)
    if function =='ExponentialTangentSigmoid':
        return ExponentialTangentSigmoid(0)
    if function =='ExponentialSigmoid':
        return ExponentialSigmoid(0)
    if function =='Average':
        return Average(0)

def fuzzy_rank(CF, top, function, penalty):
    R_L = np.zeros(CF.shape)
    for i in range(CF.shape[0]):
        for j in range(CF.shape[1]):
            for k in range(CF.shape[2]):
                if function =='DefaultGompertz':
                    R_L[i][j][k] = DefaultGompertz(CF[i][j][k])
                if function =='Gompertz':
                    R_L[i][j][k] = Gompertz(CF[i][j][k])
                if function =='ShiftGompertz':
                    R_L[i][j][k] = ShiftGompertz(CF[i][j][k])
                if function =='WeightedGompertz':
                    R_L[i][j][k] = WeightedGompertz(CF[i][j][k])
                if function =='DefaultExponential':
                    R_L[i][j][k] = DefaultExponential(CF[i][j][k])
                if function =='Sigmoid':
                    R_L[i][j][k] = Sigmoid(CF[i][j][k])
                if function =='DefaultMitscherlich':
                    R_L[i][j][k] = DefaultMitscherlich(CF[i][j][k])
                if function =='Mitscherlich':
                    R_L[i][j][k] = Mitscherlich(CF[i][j][k])

                if function =='Exponential':
                    R_L[i][j][k] = Exponential(CF[i][j][k])
                if function =='Logistic':
                    R_L[i][j][k] = Logistic(CF[i][j][k])
                if function =='NewBurr':
                    R_L[i][j][k] = NewBurr(CF[i][j][k])
                if function =='Ibrahim':
                    R_L[i][j][k] = Ibrahim(CF[i][j][k])
                if function =='Gamma':
                    R_L[i][j][k] = Gamma(CF[i][j][k])
                if function =='ExponentialGompertz':
                    R_L[i][j][k] = ExponentialGompertz(CF[i][j][k])
                if function =='ExponentialTangent':
                    R_L[i][j][k] = ExponentialTangent(CF[i][j][k])
                if function =='ExponentialTangentSigmoid':
                    R_L[i][j][k] = ExponentialTangentSigmoid(CF[i][j][k])
                if function =='ExponentialSigmoid':
                    R_L[i][j][k] = ExponentialSigmoid(CF[i][j][k])
                if function =='Average':
                    R_L[i][j][k] = Average(CF[i][j][k])

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
    #print ("K_L: ", K_L)
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

def Distributions(function, top = 2, *argv):
    L = 0 #Number of classifiers
    for arg in argv:
        L += 1
    
    num_classes = arg.shape[1]
    CF = np.zeros(shape = (L,arg.shape[0], arg.shape[1]))
    penalty = get_penalty(function)
    print ("Penalty: ", penalty)
    for i, arg in enumerate(argv):
        CF[:][:][i] = arg
    print (CF.shape)
    R_L = fuzzy_rank(CF, top, function, penalty) #R_L is with penalties
    RS = np.sum(R_L, axis=0)
    CFS = CFS_func(CF, R_L, penalty)
    FS = RS*CFS

    predictions = np.argmin(FS,axis=1)
    return predictions


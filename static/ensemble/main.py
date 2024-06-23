from utils_ensemble import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--distribution_function', type=str, default = 'Gompertz', help='Distribution function: Gompertz, ShitfGompertz')
parser.add_argument('--data_directory', type=str, default = '', help='Directory where csv files are stored')
parser.add_argument('--topk', type=int, default = 2, help='Top-k number of classes')
args = parser.parse_args()

root = args.data_directory
if not root[-1]=='/':
    root=root+'/'

p1,labels = getfile(root+"efficient_b3")
p2,_ = getfile(root+"inception_v3")
p3,_ = getfile(root+"resnet")
p4,_ = getfile(root+"densenet201")

#Check utils_ensemble.py to see the "labels" distribution. Change according to the dataset used. By default it has been set for the SARS-COV-2 dataset.

#Calculate Gompertz Function Ensemble


top = args.topk #top 'k' classes
function = args.distribution_function
print ("Distribution function: ", function)
predictions = Distributions(function, top, p1, p2, p3, p4)

print("predictions: ", predictions.shape)
print("labels: ", labels.shape)
correct = np.where(predictions == labels)[0].shape[0]
total = labels.shape[0]

print("Accuracy = ",correct/total)
classes = []
for i in range(p1.shape[1]):
    classes.append(str(i+1))
metrics(labels,predictions,classes)

plot_roc(labels,predictions)

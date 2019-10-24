import torch
import numpy as np
import torchvision
import torchvision.datasets as datasets
import math
import model as M
    

def normalize(data):
    for d in data:
        d = d - d.mean()
    return data

mnist_train = datasets.MNIST(root='C:\\Users\Justin Wang\Documents\Projects', train=True).data.float() / 255
mnist_train = normalize(mnist_train)
mnist_train_labels = datasets.MNIST(root='C:\\Users\Justin Wang\Documents\Projects', train=True).targets
mnist_val = datasets.MNIST(root='C:\\Users\Justin Wang\Documents\Projects', train=False).data.float() / 255
mnist_val = normalize(mnist_val)
mnist_val_labels = datasets.MNIST(root='C:\\Users\Justin Wang\Documents\Projects', train=False).targets

num_classes = 10

network = [M.GMN(10, [200, 100, 50, 1], 28**2, 4) for i in range(num_classes)]

break_num = 100

for i in range(min(break_num, len(mnist_train))):
    if ((i + 1)%1 == 0):
        print("Training:", 100 * (i + 1) / break_num, "% Completed")
    for j in range(len(network)):
        network[j].train_on_sample(mnist_train[i].view(-1), 1 if j == mnist_train_labels[i] else 0, min(8000 / (i + 1), 0.3))
            
val_ratio = 0.5
correct = 0
total = 0
for i in range(min(math.floor(break_num * val_ratio), len(mnist_val))):
        
    prob = torch.cat([network[j].infer(mnist_val[i]).unsqueeze(0) for j in range(len(network))])
    
    pred = torch.argmax(prob)
    if (pred == mnist_val_labels[i]):
        correct = correct + 1
    total = total + 1

print("Accuracy:", 100 * (correct / total), "%")
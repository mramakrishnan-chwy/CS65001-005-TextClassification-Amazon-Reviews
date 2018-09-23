#IMPORTNAT NOTE: #Before running the code, please run mr6rx-bag-of-words  model to have all the varibles in place

import numpy as np
import matplotlib.pyplot as plt
#import 'mr6rx-bag-of-words

#Unique y values
y_unique_len = len(set(train_label_int[:,0]))

def perceptron(theta, iterations):
    
    for each in range(iterations):
        
        #Creating the feature function for the current training sample
        feature_vector = feature_function(train_data[each],[0,1])
        y_predict = np.argmax(np.matmul(theta.T , feature_vector)[0])
        y_actual = train_label_int[each]
    
    
        #Checking if the prediction is not equal to actual
        if(y_predict != y_actual):
            #Following codes get executed if the actual is not equal to predicted
              
            f_x_y = feature_vector[:,y_actual]
            f_x_y.shape = f_x_y.shape[0],1
    
            f_x_y_pred = feature_vector[:,y_predict]
            f_x_y_pred.shape = f_x_y_pred.shape[0],1
    
            #Updating the theta value    
            theta +=  f_x_y - f_x_y_pred
    
        else:
            theta = theta
            
    return theta

def predictions(theta,input_data):
    y_predict_all = []
    for each in range(len(input_data)):
        #print(each)
        feature_vector = feature_function(input_data[each],[0,1])
        y_predict = np.argmax(np.matmul(theta.T , feature_vector)[0])
        y_actual = train_label_int[each]
        y_predict_all.append(y_predict)
        
    return(y_predict_all)
    
    
    
    
########################## RUNNING THE PERCEPTRON ALGORITHM ###################################
#Initializing theta with random values
theta = np.random.rand(len(vocab_list) * y_unique_len,1)

training_accuracy_all = []
dev_accuracy_all = []
for epochs in range(0,10):
    
    print("\n")
    print("Running epoch ",epochs+1)
    
    theta = perceptron(theta,len(train_data))
    
    print("Completed Training Epoch ",epochs+1,"...")
    print("Performing Evaluation: ")
    train_preds = predictions(theta,train_data)
    train_accuracy = np.sum(train_preds == train_label_int.T[0])/len(train_preds)
    print("Training Accuracy: ",train_accuracy)
    
    dev_preds = predictions(theta,dev_data)
    dev_accuracy = np.sum(dev_preds == dev_label_int.T[0])/len(dev_preds)
    print("Dev Accuracy: ",dev_accuracy)
    
    training_accuracy_all.append(train_accuracy)
    dev_accuracy_all.append(dev_accuracy)



# Accuracy over different epochs
plt.plot(list(range(1,11)),training_accuracy_all,'r--', list(range(1,11)), dev_accuracy_all)
plt.ylabel('Prediction Accuracy')
plt.show()

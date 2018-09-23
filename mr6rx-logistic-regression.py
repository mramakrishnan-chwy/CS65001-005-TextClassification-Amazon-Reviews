#IMPORTNAT NOTE: Before running the code, please run mr6rx-bag-of-words  model to have all the varibles in place

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# train_tokenize = [nltk.word_tokenize(each) for each in train_data]
# dev_tokenize = [nltk.word_tokenize(each) for each in dev_data]

#### Training Labels
#Correcting the shape of labels
train_label_int.shape = (train_label_int.shape[0],)
dev_label_int.shape   = (dev_label_int.shape[0],)

### PART  1. Using Default settings of count vectorizer
cv_vectors = CountVectorizer(train_data)
cv_word_vectors = cv_vectors.fit_transform(train_data)

train_vectors = cv_word_vectors


lr_model = LogisticRegression()
lr_model.fit(train_vectors,train_label_int)

######### Model Evaluation - Training set ############################

training_output = lr_model.predict(train_vectors)
100 * np.sum(training_output == train_label_int)/len(training_output)

######### Model Evaluation - Dev set ############################

dev_vectors = cv_vectors.transform(dev_data)
dev_predictions = lr_model.predict(dev_vectors)
100 * np.sum(dev_predictions == dev_label_int)/len(dev_label_int)

#####  PART  2. Changing the n_gram argument in the CountVectorizer function from (1,1) to (1,2)

cv_vectors = CountVectorizer(train_data,ngram_range= (1,2))
cv_word_vectors = cv_vectors.fit_transform(train_data)


train_vectors = cv_word_vectors

lr_model_more_features = LogisticRegression()
lr_model_more_features.fit(train_vectors,train_label_int)

######### Model Evaluation - Training set ############################

training_output = lr_model_more_features.predict(train_vectors)
100 * np.sum(training_output == train_label_int)/len(training_output)

######### Model Evaluation - Dev set ############################

dev_vectors = cv_vectors.transform(dev_data)
dev_predictions = lr_model_more_features.predict(dev_vectors)
100 * np.sum(dev_predictions == dev_label_int)/len(dev_label_int)


#####  PART  3. Changing the regularization parameter lambda in the Logistic Regression function

lambda_vals = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10,100]

for lambda_val in lambda_vals:
    
    print("Lambda val: ",lambda_val)
    lr_model_lambda = LogisticRegression(C= 1/lambda_val)
    lr_model_lambda.fit(train_vectors,train_label_int)
    
    training_output = lr_model_lambda.predict(train_vectors)
    print("Training Accuracy: ",100 * np.sum(training_output == train_label_int)/len(training_output))
    
    dev_predictions = lr_model_lambda.predict(dev_vectors)
    print("Dev Accuracy: ",100 * np.sum(dev_predictions == dev_label_int)/len(dev_label_int))   
    print("\n")
    


## Based on the previous Analysis we choose a narrower range for lambda 
    
lambda_vals_new = [1,5,10,15,20,30,50]    

for lambda_val in lambda_vals_new:
    
    print("Lambda val: ",lambda_val)
    lr_model_lambda = LogisticRegression(C= 1/lambda_val)
    lr_model_lambda.fit(train_vectors,train_label_int)
    
    training_output = lr_model_lambda.predict(train_vectors)
    print("Training Accuracy: ",100 * np.sum(training_output == train_label_int)/len(training_output))
    
    dev_predictions = lr_model_lambda.predict(dev_vectors)
    print("Dev Accuracy: ",100 * np.sum(dev_predictions == dev_label_int)/len(dev_label_int))   
    print("\n")
    
    
####  PART  4. Explaining L1 regularization
    
#1. L1 Regularization
    
'''
 L1 regularizaion uses a slightly different loss function when compared to L2. i.e. L1 uses a 
 lambda times modulus of theta along with the regular loss function in logistic regression.
 
 So, now the countour plot for L1 looks like the image uploaded 'L1 vs L2.png'
 
 It can be noted that theta will generally move towards the center point of the contour circle (without the presence of regularization),
 but with the presence of L1 regularization, we have contours for it in the shape (diamond) as shown for
 the L1 picture. 
 
 So, the theta will try to move towards the centers of both the contours based on the weight. However,
 it will settle inbetween the two regions and it is seen that the nearest point for L1 contour lies 
 on w2 axis which means w1 is zero. Therefore while trying to settle inbetween the two regions, theta will
 find an optimal point in the w2 axis because of which w1 will turn out to be zero.
 
 Thus at higher dimensions similar situation happends and many other weights become zero and hence
 we end up with a sparse solution as compared to L2.
 
  
'''

#2. Different lambda values

print("Trying L1 Regularization")
lambda_vals_l1 = [10**-1,1,5,8,10,15,20,30,50]    

for lambda_val in lambda_vals_l1:
    
    print("Lambda val: ",lambda_val)
    lr_model_lambda = LogisticRegression(C= 1/lambda_val,penalty="l1")
    lr_model_lambda.fit(train_vectors,train_label_int)
    
    training_output = lr_model_lambda.predict(train_vectors)
    print("Training Accuracy: ",100 * np.sum(training_output == train_label_int)/len(training_output))
    
    dev_predictions = lr_model_lambda.predict(dev_vectors)
    print("Dev Accuracy: ",100 * np.sum(dev_predictions == dev_label_int)/len(dev_label_int))   
    print("\n")



#####  PART  5. Explaining L1 regularization

#### Improving the Model Accuracy
# 1. Richer Feature set
# Using tri-gram and four gram word vectors    
# 2. Trying different regularization parameter #l1 vs l2 and different lambda values
# 3. Trying different optimization techinques in logistic regression    

cv_vectors = CountVectorizer(train_data,ngram_range=  (1,4), min_df = 2)
cv_word_vectors = cv_vectors.fit_transform(train_data)

train_vectors = cv_word_vectors
dev_vectors = cv_vectors.transform(dev_data)
   
#lambda_vals_new = [1,2,3,4,5,6,7,10]    
lambda_vals_new = [1,3,4,5,7,6,8,9]    

for lambda_val in lambda_vals_new:
    
    print("Lambda val: ",lambda_val)
    lr_model_lambda = LogisticRegression(C= 1/lambda_val)
    lr_model_lambda.fit(train_vectors,train_label_int)
    
    training_output = lr_model_lambda.predict(train_vectors)
    print("Training Accuracy: ",100 * np.sum(training_output == train_label_int)/len(training_output))
    
    dev_predictions = lr_model_lambda.predict(dev_vectors)
    print("Dev Accuracy: ",100 * np.sum(dev_predictions == dev_label_int)/len(dev_label_int))   
    print("\n")


####### BEST MODEL 

from sklearn.feature_extraction.text import TfidfVectorizer    
cv_vectors = CountVectorizer(train_data,ngram_range=  (1,4), min_df = 2)
cv_word_vectors = cv_vectors.fit_transform(train_data)

train_vectors = cv_word_vectors
dev_vectors = cv_vectors.transform(dev_data)

#Setting lambda value for l2 as 5
lambda_val = 5

print("Lambda val: ",lambda_val)
lr_model_lambda = LogisticRegression(C= 1/lambda_val)
lr_model_lambda.fit(train_vectors,train_label_int)

training_output = lr_model_lambda.predict(train_vectors)
print("Training Accuracy: ",100 * np.sum(training_output == train_label_int)/len(training_output))

dev_predictions = lr_model_lambda.predict(dev_vectors)
print("Dev Accuracy: ",100 * np.sum(dev_predictions == dev_label_int)/len(dev_label_int))   
print("\n")
    

################ TEST PREDICTION UING THE BEST MODEL #####################

test_vectors = cv_vectors.transform(test_data)
test_predictions = lr_model_lambda.predict(test_vectors)

np.savetxt('mr6rx-lr-test.pred.txt',test_predictions,fmt = '%1.0i')




   
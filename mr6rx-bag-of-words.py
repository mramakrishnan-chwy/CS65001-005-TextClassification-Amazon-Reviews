import numpy as np
import re
os.chdir("C:\\Users\\arvra\\Documents\\UVa files\\Classes\\Fall_18\\NLP\\Projects\\Project 1")

######### Reading the Input data: #############################
## Reading the training data and its labels
#Train data
train_file = open('trn.data.txt', 'r')
train_data = train_file.readlines()
train_file.close()  

#Train label
train_lab_file = open('trn.label.txt', 'r')
train_label = train_lab_file.readlines()
train_lab_file.close()

## Reading the dev data and its labels

#Dev data
dev_file = open('dev.data.txt', 'r')
dev_data = dev_file.readlines()
dev_file.close()

#Dev label
dev_lab_file = open('dev.label.txt', 'r')
dev_label = dev_lab_file.readlines()
dev_lab_file.close()


## Reading the test data

#Test data
test_file = open('tst.data.txt', 'r')
test_data = test_file.readlines()
test_file.close()

#### Cleaning the response variable file for train and dev datasets 

train_label_int = np.array([int(each.strip()) for each in train_label])
dev_label_int = np.array([int(each.strip()) for each in dev_label])

#Cleaning the shape for the train and dev dataset
train_label_int.shape = (30000,1)
dev_label_int.shape = (10000,1)

###Converting all the text into lower case and removing special characters and unwanted text
train_data = [re.sub('[^a-zA-Z0-9 \n]', '',each.lower().strip()) for each in train_data]
dev_data = [re.sub('[^a-zA-Z0-9 \n]', '',each.lower().strip()) for each in dev_data]
test_data = [re.sub('[^a-zA-Z0-9 \n]', '',each.lower().strip()) for each in test_data]


######### Creating Vocabulary #############################
#Creating the vocabulary from the training dataset 
Entire_vocab = " ".join(train_data)
all_words = [each for each in Entire_vocab.split(" ") if each != '']

#Creating a dictionary to have the word counts for each word in the document
dict_word_counts  = {}
    
for each in all_words:

    if(each not in dict_word_counts.keys()):
        dict_word_counts[each] = 1
        
    else:
        dict_word_counts[each] += 1

#Now removing the words with very low frequency
min_frequency = 3
vocab_list = [key for key, value in dict_word_counts.items() if value > min_frequency]
len(vocab_list)
vocab_list.sort()

#This is the entire vocabulary size
f_x_dict = {}
    
#Creating a dictionary to store the position of each element of the vocabulary
for value,each in enumerate(vocab_list):
    f_x_dict[each] = value
 
    
######### Feature Function #############################    
#x - input sentence
#y - list of all the output labels  
        
def feature_function(x,y):
    
    final_feature_representation = []
    
    #Looping through the length of y (Loops twice if y - (0,1))
    for each in range(len(y)):
        
        feature_vector_placeholder = np.zeros(len(vocab_list)*len(y))
        
        #Getting the word positions in the vocabulary list for each sentence
        word_positions = np.array([f_x_dict[each] for each in x.split(" ") if each in f_x_dict.keys()])
        
        
        #Updating the feature_vector_placeholder with the word positons we obtain
        if(len(word_positions)>0):
            
            #Now, making the bow of words features for the current sentence
            #Based on the y value, we make the corresponding features as non zero
            #Say if y=0, we just make <word1,0> and <word2,0>,... as non zero
            #If y = 1, we make <word1,1> and <word2,1>,... as non zero
            #This is governed by the (each * len(vocab_list)) part of the indexing below

            for each_pos in word_positions + (each * len(vocab_list)):
                #print(each_pos)
                feature_vector_placeholder[each_pos] += 1
            
            
        #Chaning the shape of the vector as required
        feature_vector_placeholder.shape = len(feature_vector_placeholder),1
        
            
        if(each == 0):
            
            final_feature_representation = feature_vector_placeholder
            
        else:
            
            final_feature_representation = np.concatenate((final_feature_representation,feature_vector_placeholder),axis = 1)
    
    
    return(final_feature_representation)
    

#Example
#Sample index - BOW model
index = 1    
bow_rep = feature_function(train_data[index],[0,1])    
bow_rep



import pandas as pd
import numpy as np
import copy
import sys, os

import tensorflow as tf

from tensorflow.python.keras.layers import Input, Dense, Activation, Flatten, Dropout, Embedding, concatenate
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.optimizers import Adadelta
from keras.regularizers import l1

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

df = pd.read_csv(os.path.join(__location__, 'train.csv'))  
df2 = pd.read_csv(os.path.join(__location__, 'test.csv'))

y = df['target']   # the taget variable 
df = df.drop(['target','id'],axis = 1)    
df2 = df2.drop(['id'],axis = 1)

df = df.astype('category')   #converting to pandas' categorical variables to be used below for encoding 
df2 = df2.astype('category')  #converting to pandas' categorical variables to be used below for encoding 

DF = copy.deepcopy(df)
DF2 = copy.deepcopy(df2)

for x in list(df):
    d1 = dict(enumerate(df[x].cat.categories ))    # conerting the index from categories to dictionary
    in_d1 = dict([[v,k] for k,v in d1.items()])    # we actually need the inverse for mapping
    DF[x] = df[x].map(in_d1)                       # mapping the training set
    DF2[x] = df2[x].map(in_d1)                     # mapping the submission set



DF_train, DF_test, y_train, y_test = train_test_split(DF, y, test_size=0.05)   # splitting to training and test set 
 
# some of the categories in the submission set are not in the training set so they need to be mapped to the dummy index
 
DFmax = DF_train.max()   
DF2  = DF2.fillna(-1)   #all the categories not in the training set are mapped to NaN
for x in list(DF2):
    DF2.loc[DF2[x] ==-1, x] = (DFmax[x]+1).astype(int)

### Below is the model and the training of the model done in keras  

emb_input = [Input(shape = (1,)) for i in range(DF_train.shape[1])]   

reg1 = l1(.00004)    #regularizer for the embedding layer
reg2 = l1(.000002)   #regularizer for the dense layers

in_size = (DF_train.max(axis=0)+2).values.astype(int)  #The number of unique values in the catogorical columns
out_size = np.rint(in_size**.25).astype(int)  # the output dimension of the embedding layers

emb_list = [Embedding(output_dim=out_size[i], input_dim=in_size[i], input_length=1, embeddings_regularizer=reg1)(emb_input[i]) for i in range(DF_train.shape[1])]

emb = concatenate(emb_list)

emb = Flatten()(emb)

x = Dense(100, activation='relu',kernel_regularizer = reg2)(emb)
x = Dropout(.7)(x)
x = Dense(100, activation='relu',kernel_regularizer = reg2)(x)
x = Dropout(.7)(x)
x = Dense(1, activation='sigmoid',kernel_regularizer = reg2)(x)


model = Model(inputs = [emb_input[i] for i in range(DF_train.shape[1])],outputs = [x])

model.summary()

optimizer = Adadelta(lr=3.0, rho=0.95, epsilon=None, decay=0.0)

model.compile(optimizer=optimizer, loss=['binary_crossentropy'])

model.fit([DF_train[i] for i in list(DF_train)],
                     y_train,epochs=60,batch_size = 4096,shuffle=True)
                     
Y_pred = model.predict([DF_train[i] for i in list(DF_train)])
y_pred = model.predict([DF_test[i] for i in list(DF_train)])

print(roc_auc_score(y_test, y_pred))
print(roc_auc_score(y_train, Y_pred))
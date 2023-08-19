# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Here the basic neural network model has been created with one input layer, one hidden layer and one output layer.The number of neurons(UNITS) in each layer varies the 1st input layer has 16 units and hidden layer has 8 units and output layer has one unit.
In this basic NN Model, we have used "relu" activation function in input and hidden layer, relu(RECTIFIED LINEAR UNIT) Activation function is a piece-wise linear function that will output the input directly if it is positive and zero if it is negative.

## Neural Network Model

<img width="780" alt="teory" src="https://github.com/Bhavishya203/basic-nn-model/assets/94679395/8d9b71cc-f467-42a0-a2b3-c359618d7ad8">


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
## Importing modules

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## Authenticate & Create data frame using data in sheets

from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Ex1').sheet1
data = worksheet.get_all_values()
dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype({'input':'float'})
dataset1=dataset1.astype({'output':'float'})
dataset1.head()

## Assign X & Y Values

X = dataset1[['input']].values
y = dataset1[['output']].values
X

## Normalize the values and split the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
## Create a neural network and train it.
ai_brain=Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(X_train1,y_train,epochs=200)

## Plot the loss

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

## Predict for some value

X_test1 = Scaler.transform(X_test)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```

## Dataset Information
<img width="128" alt="exp1 fig 1" src="https://github.com/Bhavishya203/basic-nn-model/assets/94679395/f9576b30-94a0-420b-ae0f-8909e103bc11">

## OUTPUT

### Training Loss Vs Iteration Plot

<img width="420" alt="dp ex1 fig 1" src="https://github.com/Bhavishya203/basic-nn-model/assets/94679395/7345ff5b-9f43-49ac-a421-ae945513af65">


### Test Data Root Mean Squared Error

<img width="404" alt="dp fig3" src="https://github.com/Bhavishya203/basic-nn-model/assets/94679395/c7c72038-a6d5-409b-95f3-a62e67e4dec7">


### New Sample Data Prediction

<img width="305" alt="fig4" src="https://github.com/Bhavishya203/basic-nn-model/assets/94679395/28333b61-db04-4edc-ad7a-e9d5fd849856">


## RESULT
A Basic neural network regression model for the given dataset is developed successfully.

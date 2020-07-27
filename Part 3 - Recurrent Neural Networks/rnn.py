# Recurrent Neural Network   - Only numpy arrays can be the input for RNN networks specifically Time Series Analysis



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Importing the training set   
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values    #The .values makes it a numpy array

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) #We determined between 0 and 1 based on the normalization feature scaling which can not be greater than 1
training_set_scaled = sc.fit_transform(training_set)  #Normalized training set

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0]) #The last 60 day stock prcies
    y_train.append(training_set_scaled[i, 0]) #Stock price to be predicted t + 1
X_train, y_train = np.array(X_train), np.array(y_train) #Making x_train and y_train an num py array so that we can feed it to the neural network

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #Whenver we want to add new dimensions aka features, we need to add reshape it using numpy.reshape
# X_train.shape[0] gives us the row number/number of stock prices and X_train.shape[1], 1) gives us the number of columns/time steps and last is the number of predictors 1


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN - Named regressor since we are prediciting a continous value and classification is about predicting a class. 
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation - avoid overfitting
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) # input shape only contains the time series and number of indicators for the shape. WE don not need to specify the stock shape
regressor.add(Dropout(0.2)) #20% of neurons will be dropped off during each iteration of training

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True)) #Keeping the neuron count at 50 will add high dimensionality to our model to handle the complexity of the problem
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') 

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) #Batch size of 32 tells us that we will be backprograting into the system at every 32 google stock prices 


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) #Horizantal Concatenation we use 1 and vertical concatenation we use 0
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values  
inputs = inputs.reshape(-1,1)  #Since we did not get the input using a numpy iloc, we must reshape it so that it can be in numpy format
inputs = sc.transform(inputs)  #Recurrent netowrk was trained on the scaled values, so we must scale the inputs. ** Scale inputs and not the actual test values because we need to keep the test values the way they are. Also not using fit_transform since our ogject was already fitted to the training set. Using transfrom to scale the inputs
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()





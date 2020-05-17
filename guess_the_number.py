import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)
#this is to activate the documentaton to display the error messages

#loading and knowing more about the data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print("length of the test data= ",len(x_test))
print("length of the train data= ",len(x_train))
#knowing the shape of the data
print("the shape of the train featutes",x_train.shape)
print("the shape of the train labels",y_train.shape)
print("the shape of the test featutes",x_test.shape)
print("the shape of the test lables",y_test.shape)

#smaple image
plt.imshow(x_train[0],cmap='binary')
plt.show()

#our target variable is a catogorical variable so it needs to be encoded
y_train_encoded=to_categorical(y_train)
y_test_encoded=to_categorical(y_test)

#our neural network should have 28*28=784 input nodes and 10 output nodes and 2 hidden layers

#out input array should be vector so we need to reshape it
x_train_un=np.reshape(x_train,(60000,784))
x_test_un=np.reshape(x_test,(10000,784))
print(x_train_un.shape)
print(x_test_un.shape)

#normalizing our feature variables in train and test data
x_train_mean=np.mean(x_train_un)
x_train_std=np.std(x_train_un)
x_test_mean=np.mean(x_test_un)
x_test_std=np.std(x_test_un)
x_train_norm=(x_train_un-x_train_mean)/x_train_std
x_test_norm=(x_test_un-x_test_mean)/x_test_std

#creating dense layers (every node is connected to every other)
model=Sequential(
[
    Dense(128,activation='relu',input_shape=(784,)),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
]
)

#compiling the model and printing the summary
model.compile(
    optimizer='sgd',#stochistic gradent descent
    loss='categorical_crossentropy',
)
print(model.summary())

#fitting the model 
model.fit(x_train_norm,y_train_encoded,epochs=3)
#3 epochs indictes the whole data is trained 3 times

#evaluating the model on test data
loss=model.evaluate(x_test_norm,y_test_encoded)
print(loss)

#predicting the results using the model
pred=model.predict(x_test_norm)

#ploting the results
start_index=100
plt.figure(figsize=(12,12))
for i in range(16):
    plt.subplot(4,4,i+1)
    pre=np.argmax(pred[start_index+i])
    gt=y_test[start_index+i]
    plt.xticks([])
    plt.yticks([])
    col='g'
    if gt!=pre:
        col='red'
    plt.xlabel('predicted={} actual={}'.format(pre,gt),color=col)
    plt.imshow(x_test[start_index+i],cmap='binary')
plt.show()

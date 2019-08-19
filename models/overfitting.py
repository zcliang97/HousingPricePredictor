import matplotlib.pyplot as plt
# data preprocessing
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# neural network
from keras.models import Sequential
from keras.layers import Dense

# read in the dataset
dataset = pd.read_csv("housepricedata.csv").values

# isolate input features
X = dataset[:,0:10]
# scales dataset so that all values are between 0 and 1
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# isolate output prediction
Y = dataset[:,10]

# split the data into training (70%) data, val data (15%), and test data (15%)
X_train, X_rest, Y_train, Y_rest = train_test_split(X_scaled, Y, test_size=0.3)
# validation data is used to validate how the model is doing at each point in the training
X_val, X_test, Y_val, Y_test = train_test_split(X_rest, Y_rest, test_size=0.5)

# define the model and its layers
# regularizer tells keras to add L2 regularization by including the squared values of those parameters in
#       the overall loss function and weigh them by 0.01
# adding dropout gives a 0.3 chance that neurons are dropped out
model = Sequential([
    Dense(1000, activation='relu', input_shape=(10,)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1, activation='sigmoid'),
])

# define the loss function and optimizer
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train the data and save the history
hist = model.fit(X_train, Y_train,
                 batch_size=32, epochs=100,
                 validation_data=[X_val, Y_val])

print("Completed training")
output = model.evaluate(X_test, Y_test)[1]
print(output)

plt.figure(1)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Accuracy Progression')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.figure(2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss Progression')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
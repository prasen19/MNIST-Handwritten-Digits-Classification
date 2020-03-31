import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, LeakyReLU, Input, Dropout, Dense, Add, Dropout
from tensorflow.keras import Model, datasets, models

# Download and splliting MNIST dataset

mnist = datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

class_names = ['0','1','2','3','4','5','6','7','8','9']

# size of Image
H = x_train.shape[1]
W = x_train.shape[2]

# Adding third dimension to each image as keras takes image with 3 dimensions and 4th dimension is batch size.

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Noramalization of data between 0 to 1
x_train = x_train / 255
x_test  = x_test / 255


# Convolutional Neural Network


init = tf.random_normal_initializer(0.,0.2)

def mnist():

    I = Input(shape=[H,W,1])
        
    C1 = Conv2D(64, (11,11), padding='same', kernel_initializer=init)(I)
    L1 = LeakyReLU()(C1)
        
    C2 = Conv2D(64,(3,3), padding='same', strides=(2,2), kernel_initializer=init)(L1)
    B2 = BatchNormalization()(C2)
    L2 = LeakyReLU()(B2)
        
    C3 = Conv2D(128,(3,3), padding='same', strides=2, kernel_initializer=init)(L2)
    B3 = BatchNormalization()(C3)
    L3 = LeakyReLU()(B3)
    D3 = Dropout(0.5)(L3)
        
    C4 = Conv2D(128, (3,3), padding='same',strides=(2,2), kernel_initializer=init)(D3)
    B4 = BatchNormalization()(C4)
    L4 = LeakyReLU()(B4)
    D4 = Dropout(0.5)(L4)
        
    C5 = Conv2D(128,(3,3), padding='same', kernel_initializer=init)(D4)
    B5 = BatchNormalization()(C5)  
    L5 = LeakyReLU()(B5)
      
        
    F7 = Flatten()(L5)   
    DE7 = Dense(128)(F7)
    D7 = Dropout(0.5)(DE7)
        
    DE8 = Dense(64)(D7)
        
    out = Dense(10, activation='softmax')(DE8)
        
    model = Model(inputs=I, outputs=out)
    
    return model    

# Showing complete network architecture 

model = mnist()
model.summary()

model.compile(optimizer='adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train = model.fit(x=x_train, y=y_train,validation_split=0.1, batch_size=100, epochs=1) 

# Plots to display loss and accuracy

plt.figure()
plt.plot(train.history['accuracy'])
plt.plot(train.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure()
plt.plot(train.history['loss'])
plt.plot(train.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pred = model.predict(x_test)

# Showing testing results with Predicted abd actual labels 
for i in range(10):
    plt.figure()
    plt.imshow(np.squeeze(x_test[i]))
    plt.xlabel('Actual: ' + class_names[int(y_test[i])])
    plt.title('Predicted: ' + class_names[np.argmax(pred[i])])
    plt.show()
    
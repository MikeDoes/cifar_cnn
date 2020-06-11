import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import losses
def scale(set1):
    for i in range(len(set1[0])):
        for j in range(len(set1[0][0])):
            for k in range(len(set1[0][0][0])):
                set1[0][i][j][k]=np.interp(np.array(set1[0][i][j][k]), (0, 255), (0, 1))
    return set1

def one_hot(set1):
    new_set=np.zeros([len(set1),10],dtype='int8')
    for i in range(len(set1)):
        a=np.zeros(10, dtype='int8')
        a[set1[i][0]]=1
        new_set[i]=a
    return new_set

data=tf.keras.datasets.cifar10.load_data()
#Data 0 is the training and Data 1 is in test. Also the first part is the image, the second is the label in the 
indices = np.random.permutation(len(data[0][0]))
#validation_set[0][0][0][0][0] the indices represent: image number, width, height?, rgb colors, specific rgb color

validation_indices=indices[:1000]
training_indices=indices[1000:]
validation_set=[np.array(data[0][0][validation_indices],dtype='float16')/255,one_hot(data[0][1][validation_indices])]
training_set=[np.array(data[0][0][training_indices],dtype='float16')/255,one_hot(data[0][1][training_indices])]
test_set=[np.array(data[1][0],dtype='float16')/255,one_hot(data[1][1])]
#Scale the data

def main():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))

    sgd=optimizers.SGD(learning_rate=0.001)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    #Learning Rate is 10^-3, Batch is 32, epochs is 50
    history = model.fit(training_set[0], training_set[1], epochs=50, batch_size=32,
                        validation_data=(validation_set[0], validation_set[1]))
    
    main()

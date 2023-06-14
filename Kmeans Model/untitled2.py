'''
    @project: Hippocampus Segmentation
    @author: Sami Rajichi

'''
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from collections import Counter
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

class Hippocampus(object):
    
    def __init__(self):
        ''' Preparing data '''
        
        self.main_path = 'D:\Sami\Sami\ML\Task04_Hippocampus'
        self.train_dir = os.path.join(self.main_path, 'Train')
        self.label_dir = os.path.join(self.main_path, 'Labels')
        self.train_npy__dir = os.path.join(self.main_path, 'Train_npy')
        self.GT_npy__dir = os.path.join(self.main_path, 'GroundTruth_npy')
        self.train = [img for img in sorted(os.listdir(self.train_dir)) if 'hippocampus' in img]
        self.labels = [img for img in sorted(os.listdir(self.label_dir)) if 'hippocampus' in img]
        self.X = list()
        self.Y = list()
        self.img_mean_shape = None
    
    def loadingAndAnalysingData(self):
        ''' Load training and labeling data after keeping track of all the irrelevant and distorted data '''
        
        self.train_set = [img for img in self.train if img in self.labels]
        self.GT_set = [img for img in self.labels if img in self.train]
        print(len(self.train), len(self.labels))
        print(len(self.train_set), len(self.GT_set))
        print(self.train_set,'\n')
        
        #return self.train_set, self.GT_set
    
    def substitutingToNPY(self):
        ''' Modulating both of the training files as well as the ground-truth files to npy arrays '''
        
        #self.X = list()
        #self.Y = list()
        for img_t, img_gt in zip(self.train_set, self.GT_set):
            t = nib.load(os.path.join(self.train_dir, img_t)).get_data()
            gt = nib.load(os.path.join(self.label_dir, img_gt)).get_data()
            t = np.array(t)
            gt = np.array(gt)
            self.X.append(t)
            self.Y.append(gt)
        print(self.X[0])
        self.X = np.array(self.X, dtype=np.float32)
        print(self.X[0])
        self.Y = np.array(self.Y, dtype=np.float32)
        #self.X = tf.convert_to_tensor(self.X)
        #self.Y = tf.convert_to_tensor(self.Y)
        
        return self.X[0], self.Y[0]
    
    def _save(self):
        ''' Saving each of the generated npy arrays in a separated directories '''
        
        k = 0
        for img_t, img_gt in zip(self.train_set, self.GT_set):
            np.save(os.path.join(self.train_npy__dir, str(img_t).split('.')[0] + '.npy'), self.X[k])
            np.save(os.path.join(self.GT_npy__dir, str(img_gt).split('.')[0] + '.npy'), self.Y[k])
            k += 1
        print('*** Saving process done with success *** \n')
        
    def getMeanSize(self):
        ''' Computing the average size among all images sizes '''
        
        dct = {
            'x': Counter([img.shape[0] for img in self.X]).most_common()[0][0],
            'y': Counter([img.shape[1] for img in self.X]).most_common()[0][0],
            'c': Counter([img.shape[2] for img in self.X]).most_common()[0][0]}
        width, height, channel = dct['x'], dct['y'], dct['c']
        self.img_mean_shape = (width, height, channel)
        print('Average shape =', self.img_mean_shape)
        
        return self.img_mean_shape
        
    def resizingData(self):
        ''' Resizing each image shape to a unique one '''
        
        for i in range(len(self.X)):
            self.X[i] = np.resize(self.X[i], self.img_mean_shape)
            self.Y[i] = np.resize(self.Y[i], self.img_mean_shape)
        
        for i in range(5):
            print(self.X[i].shape)
            print(self.Y[i].shape)
            
        
    def displayTrainingSet(self, nb_samples=5, slice_index=25):
        ''' Displaying n models of Hippocampus image along with their labels (or ground-truth) '''
        
        plt.figure(figsize= (5, 25), dpi = 80)
        k = 0
        for i in range(nb_samples):
            k += 1
            plt.subplot(12, 2, k)
            plt.imshow(self.X[i][:,:,slice_index])
            plt.title('Original')
            plt.subplots_adjust(wspace = 0)
            plt.axis('off')
            k += 1
            plt.subplot(12, 2, k)
            plt.imshow(self.Y[i][:,:,slice_index])
            plt.title('Ground Truth')
            plt.subplots_adjust(wspace = 0)
            plt.axis('off')
            
    def normalizing(self, dataset):
        ''' Scaling input vectors individually to unit norm (vector length) '''
        
        for i in range(dataset.shape[0]):
            dataset[i] = normalize(dataset[i], norm='max', copy=True, return_norm=False)
            
        return dataset
    
    def STNData(self):
        ''' Split, Train and Normalize data '''
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=30)
        #self.x_train = self.normalizing(self.x_train)
        #self.x_test = self.normalizing(self.x_test)
        #self.y_train = self.normalizing(self.y_train)
        
        #return self.x_train, self.x_test, self.y_train

    def buildModel(self):
        ''' I'll be using U-net model for its efficiency of segmenting biomedical images '''
        
        model = Sequential()

        # The first two layers with 32 filters of window size 3x3
    
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.img_mean_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
    
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
     
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
     
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        
        model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def fittingModel(self):
        ''' It's time to train our model '''
        
        batch_size = 15
        epochs = 30
        
        model = Hippocampus.buildModel(self)
        his = model.fit(self.x_train, self.y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size)
        return his
    
    def modelAccuracy(self):
        ''' Visualize the model's accuracy '''
        
        history = Hippocampus.fittingModel(self)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Valiation'], loc='upper left')
        plt.show()
        
    def modelLoss(self):
        ''' Visualize the model's loss '''
        
        plt.plot(self.his.his['loss'])
        plt.plot(self.his.his['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Valiation'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    hippocampus = Hippocampus()
    hippocampus.loadingAndAnalysingData()
    print(hippocampus.substitutingToNPY())
    '''img_mean_shape = hippocampus.getMeanSize()
    hippocampus.resizingData()
    #hippocampus._save()
    #hippocampus.displayTrainingSet()
    hippocampus.STNData()
    hippocampus.buildModel().summary()
    hippocampus.modelAccuracy()'''
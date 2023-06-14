import  nibabel as nib
import os
from skimage.io import imshow
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv3D, MaxPooling3D, Flatten
import pandas as pd

path_main = 'D:\Sami\Sami\ML\Task04_Hippocampus'

image_path_main = os.path.join(path_main, 'Train')

label_pathmain = os.path.join(path_main, 'Labels')

img_paths = [img for img in sorted(os.listdir(image_path_main)) if 'nii' in img]
print(len(img_paths))

label_paths = [img for img in sorted(os.listdir(label_pathmain)) if 'nii' in img]
print(len(label_paths))

X = []
Y = []
for img in img_paths:
    if img in label_paths:
        X.append(img)

for img in label_paths:
    if img in img_paths:
        Y.append(img)
        
print(len(X))
print(len(Y)

img = nib.load(os.path.join(image_path_main, X[1])).get_fdata()


data = pd.DataFrame(dataset)
    
print(data.head())

data = pd.DataFrame(dataset)
    
print(data.head())

def create_model(img):

    model = Sequential()

    # The first two layers with 32 filters of window size 3x3

    model.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu', input_shape=img.shape))
    model.add(Conv3D(32, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model



# The main role of show_image is to display image out on the screen using matplotlib module
def show_image(num_image, im, lb):
    '''
    This function contains 3 parameters:
        num_image: index of the image in the array 
        im: array of images 
        lb: array of labels
    '''
    img = nib.load(os.path.join(image_path_main, im[num_image])).get_fdata()
    label = nib.load(os.path.join(label_pathmain, lb[num_image])).get_fdata()
    print(img.shape)
    print(label.shape)
    
    print('Image Min-Max values: Image={},{} and label={},{}'.format(img.max(), img.min(), label.max(), label.min()))
    print('Number of subclasses = ', int(label.max())+1)
    
    ax = plt.subplot('121')
    ax.imshow(img[:,:,10], cmap='gray')
    ax.set_title('Input image')
    ax = plt.subplot('122')
    ax.imshow(label[:,:,10], cmap='gray')
    ax.set_title('Segmentation Mask')
    
    plt.show()
    print('\n')

# Testing show_image function with the first image 
show_image(0, X, Y)

# Testing show_image function with 10 images in a row 
for i in range(10):
    show_image(i, X, Y)

def show_segmented_image(num_image, X, Y):
    
    img = nib.load(os.path.join(image_path_main, X[num_image])).get_fdata()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X[num_image], Y[num_image], test_size=0.2, random_state=20)
    model = create_model(img)
    model.fit(X_train, y_train)
    img_seg = model
    print(img.shape)
    print(img_seg.shape)
    
    ax = plt.subplot('121')
    ax.imshow(img[:,:,10], cmap='gray')
    ax.set_title('Input image')
    ax = plt.subplot('122')
    ax.imshow(img_seg[:,:,10], cmap='gray')
    ax.set_title('Segmented Image')
    
    plt.show()
    print('\n')

# Testing show_segmented_image function with the first image 
show_segmented_image(0, X, Y)

# Testing show_segmented_image function with 10 images in a row 
for i in range(10):
    show_segmented_image(i, X, Y)
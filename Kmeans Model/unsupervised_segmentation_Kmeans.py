import  nibabel as nib
import os
from skimage.io import imshow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, chi2
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
print(len(Y))
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)
'''
#print(len(x_train), len(x_test), len(y_train), len(y_test))

def extract_features(img):
    #apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(img)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features

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
    extract_features(img)

# Testing show_image function with the first image 
#show_image(0, x_train, y_train)

imageseg_path_main = os.path.join(path_main, 'npy_X')

labelseg_pathmain = os.path.join(path_main, 'npy_Y')

for img in X:
    image = nib.load(os.path.join(image_path_main, img)).get_data()
    image = np.array(image)
    np.save(os.path.join(imageseg_path_main, str(img).split('.')[0] + '.npy'), image)
    
for img in Y:
    image = nib.load(os.path.join(label_pathmain, img)).get_data()
    image = np.array(image)
    np.save(os.path.join(labelseg_pathmain, str(img).split('.')[0] + '.npy'), image)
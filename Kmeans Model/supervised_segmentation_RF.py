import  nibabel as nib
import os
from skimage.io import imshow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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

dataset = {
    'Images': [],
    #'Roberts': [],
    #'Sobel': [],
    #'Scharr': [],
    #'Prewitt': [],
    #'Gaussian s3': [],
    #'Gaussian s7': [],
    'Labels': []
    }

img = nib.load(os.path.join(image_path_main, X[1])).get_fdata()


data = pd.DataFrame(dataset)
    
print(data.head())

def extract_features(df, X, Y, index):
    
    img = nib.load(os.path.join(image_path_main, X[index])).get_fdata()
    img2 = img.reshape(-1, img.shape[2])
    df['Images'].append(img2.shape[0])
    
    img = nib.load(os.path.join(label_pathmain, Y[index])).get_fdata()
    img2 = img.reshape(-1, img.shape[2])
    #print(img2.shape[2])
    df['Labels'].append(img2.shape[0])
    '''
    from skimage.filters import roberts, sobel, scharr, prewitt
    
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'].append(edge_roberts1.shape[0])
    
    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'].append(edge_sobel1.shape[0])
    
    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'].append(edge_scharr1.shape[0])
    
    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'].append(edge_prewitt1.shape[0])
    
    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'].append(gaussian_img1.shape[0])
    
    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'].append(gaussian_img3.shape[0])
    '''
    return df

dataset = extract_features(dataset, X, Y, 0)
data = pd.DataFrame(dataset)
    
print(data.head())

def random_forest(data):
    
    x = data['Images'].values
    y = data.drop(labels=['Images'], axis=1).values
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(x_train, y_train)

    return rfc


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
    '''
    This function contains 2 parameters:
        num_image: index of the image in the array 
        im: array of images 
    '''
    img = nib.load(os.path.join(image_path_main, X[num_image])).get_fdata()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X[num_image], Y[num_image], test_size=0.2, random_state=20)
    img_seg = random_forest(data)
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
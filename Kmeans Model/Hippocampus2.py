import  nibabel as nib
import os
from skimage.io import imshow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

path_main = 'D:\Sami\Sami\ML\Task04_Hippocampus'

image_path_main = os.path.join(path_main, 'Train')

label_pathmain = os.path.join(path_main, 'Labels')

img_paths = [img for img in sorted(os.listdir(image_path_main)) if 'nii' in img]
print(len(img_paths))

label_paths = [img for img in sorted(os.listdir(label_pathmain)) if 'nii' in img]
print(len(label_paths))

im = []
lb = []
for img in img_paths:
    if img in label_paths:
        im.append(img)

for img in label_paths:
    if img in img_paths:
        lb.append(img)
        
print(len(lb))
print(len(im))

def kmeans_model(image=''):
    img_seg = image.reshape(-1, image.shape[2])/255.0

    print(img_seg.shape)

    km = KMeans(n_clusters=2)
    km.fit(img_seg)

    img_seg = km.cluster_centers_[km.labels_].reshape(image.shape)
    img_seg = img_seg.reshape(image.shape)
    
    return img_seg


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
show_image(0, im, lb)

# Testing show_image function with 10 images in a row 
for i in range(10):
    show_image(i, im, lb)

def show_segmented_image(num_image, im):
    '''
    This function contains 2 parameters:
        num_image: index of the image in the array 
        im: array of images 
    '''
    img = nib.load(os.path.join(image_path_main, im[num_image])).get_fdata()
    img_seg = kmeans_model(img)
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
show_segmented_image(0, im)

# Testing show_segmented_image function with 10 images in a row 
for i in range(10):
    show_segmented_image(i, im)

'''
# Let's write a function to show all 3 views of the image and it's corresponding label
def show_n_slices(img_arr, lbl_arr, slices=None, aspect_ratio = [1, 1, 1]):

    This function visualize N slices along with their labesl
    img_arr: numpy array of the image
    lbl_arr: numpy array of the label
    slices: list of the slices of interest (integers) [axial, coronal, sagittal] to visualize
    if None, would show the central slices
    aspect_ratio: list of aspect ratios for axial, coronal, sagittal ([1,1,1] is default)

    img_arr = np.flip(img_arr.T)
    lbl_arr = np.flip(lbl_arr.T)
    
    ax_slices = img_arr.shape[0]
    cor_slices = img_arr.shape[1]
    sag_slices = img_arr.shape[2]

    fig, ax = plt.subplots(2, 3, figsize=[10,6], constrained_layout=True)
    
    if slices is None:
        slice_ax_n = int(img_arr.shape[0] / 2)
        slice_cor_n = int(img_arr.shape[1] / 2)
        slice_sag_n = int(img_arr.shape[2] / 2)
    elif (slices[0] <= img_arr.shape[0]) & (slices[1] <= img_arr.shape[1]) & (slices[2] <= img_arr.shape[2]):
        slice_ax_n = int(slices[0])
        slice_cor_n = int(slices[1])
        slice_sag_n = int(slices[2])     
    else:
        return print('Wrong slices set')
    
    ax[0, 0].set_title(f'Axial slice {slice_ax_n}')
    ax[0, 0].imshow(img_arr[slice_ax_n, :, :], cmap='gray', \
                                  aspect = aspect_ratio[0], )
    ax[1, 0].set_title(f'Label for axial slice {slice_ax_n}')
    ax[1, 0].imshow(lbl_arr[slice_ax_n, :, :], cmap='gray', \
                                  aspect = aspect_ratio[0])
    
    ax[0, 1].set_title(f'Coronal slice {slice_cor_n}')
    ax[0, 1].imshow(img_arr[:, slice_cor_n, :], cmap='gray', \
                                  aspect = aspect_ratio[1])
    ax[1, 1].set_title(f'Label for coronal slice {slice_cor_n}')
    ax[1, 1].imshow(lbl_arr[:, slice_cor_n, :], cmap='gray', \
                                  aspect = aspect_ratio[1])
    
    ax[0, 2].set_title(f'Sagittal slice {slice_sag_n}')
    ax[0, 2].imshow(img_arr[:, :, slice_sag_n], cmap='gray', \
                                  aspect = aspect_ratio[2])
    ax[1, 2].set_title(f'Label for sagittal slice {slice_sag_n}')
    ax[1, 2].imshow(lbl_arr[:, :, slice_sag_n], cmap='gray', \
                                  aspect = aspect_ratio[2])    
    plt.show()
    
show_n_slices(im, lb, slices=[7,15,7])
'''
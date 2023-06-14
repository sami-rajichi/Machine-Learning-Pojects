import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
#from skimage.io import imread
import nibabel as nib

path_main = 'D:\Sami\Sami\ML\Task04_Hippocampus\Train'

#image_main_path = os.path.join(path_main, 'expressive paintings')
images = [ img for img in os.listdir(path_main)]

image = nib.load(os.path.join(path_main, 'hippocampus_001.nii')).get_data()

plt.figure(figsize=(12, 10))
plt.imshow(image)

print(image.shape)

img_seg = image.reshape(-1, 3)/255.0

print(img_seg.shape)

km = KMeans(n_clusters=2)
km.fit(img_seg)

img_seg = km.cluster_centers_[km.labels_].reshape(image.shape)

plt.figure(figsize=(12, 10))
plt.imshow(img_seg)
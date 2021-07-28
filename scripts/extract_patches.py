import cv2
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np

from glob import glob
from numpy.lib.type_check import imag
from sklearn.utils import shuffle
from tqdm import tqdm
# Constants
patch_size = 48
patch_num = 1000
patch_threshold = 25

dataset_dir = "DRIVE/"
train_dir = dataset_dir+"training/"
groundtruth_dir = train_dir+"1st_manual/"
mask_dir = train_dir+"mask/"
images_dir = train_dir+"images/"
patch_dir = train_dir+"patches/"

# Preprocessing
def preprocess(image):
    r, g, b = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(g)
    return image

def check_coord(x,y,h,w,patch_size):
    if x-patch_size/2>0 and x+patch_size/2<h and y-patch_size/2>0 and y+patch_size/2<w:
        return True
    return False

# Patches
def extract_patches(image_path, patch_num, patch_size):
    image_id = image_path.split("/")[-1].split("_")[0]
    
    image = plt.imread(image_path)
    vessel = plt.imread(groundtruth_dir+image_id+"_manual1.gif")
    mask = plt.imread(mask_dir+image_id+"_training_mask.gif")
    
    vessel = np.where(vessel>0, 1, 0)
    mask = np.where(mask>0, 1, 0)
    
    image = preprocess(image)
    image = image * mask
    
    count = 0
    index = 0
    
    points = np.where(vessel == 1)
    
    state = np.random.get_state()
    np.random.shuffle(points[0])
    np.random.set_state(state)
    np.random.shuffle(points[1])
    
    patch_images = []
    patch_vessels = []
    
    while count < patch_num and index < len(points[0]):
        x = points[0][index]
        y = points[1][index]
        if check_coord(x, y, image.shape[0], image.shape[1], patch_size):
            if np.sum(mask[x-patch_size//2:x+patch_size//2,y-patch_size//2:y+patch_size//2])>patch_threshold:
                patch_image = image[x-patch_size//2:x+patch_size//2,y-patch_size//2:y+patch_size//2]
                patch_vessel = vessel[x-patch_size//2:x+patch_size//2,y-patch_size//2:y+patch_size//2]
                
                patch_vessel = np.where(patch_vessel>0, 255, 0)
                
                patch_images.append(patch_image)
                patch_vessels.append(patch_vessel)
                
                count = count + 1
        index = index + 1
    
    for i in range(len(patch_images)):
        plt.imsave(f"{patch_dir}{image_id}_{i}_img.jpg", patch_images[i], cmap="gray")
        plt.imsave(f"{patch_dir}{image_id}_{i}_vessel.jpg", patch_vessels[i], cmap="gray")
        # break


if __name__ == "__main__":
    for img in tqdm(glob(images_dir+"*.tif")):
        extract_patches(img, patch_num, patch_size)
        
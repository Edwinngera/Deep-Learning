import cv2
import os
from random import shuffle
from tqdm import tqdm
import  numpy as np


# Store the object name you want to classify
objectName = ['cats','dogs']

# if you do not have data set, you can use this function to make one to store the data
def mkdir(newPathName):

    folder = os.path.exists(newPathName)

    if not folder:  # if the folder does not  exist, make it.
        os.makedirs(newPathName)
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There exists this folder!  ---")

# if your images name do not satisfy request, you can use this function to change the images name
# path is the path of your images stored, obname is the name of the object
def rename(path,obname):
    i = 1
    for item in os.listdir(path):
        if item.endswith('.jpg') or item.endswith('.png'):
            old_name = path + str(item)
            if item.endswith('.jpg'):
                new_name = path+obname+'.'+str(i)+'.png'
            elif item.endswith('.png'):
                new_name = path + obname + '.' + str(i) + '.jpg'
            os.rename(old_name,new_name)
            i = i + 1

# get the label of image
def label(name):
    if name ==objectName[0]:
        tag = np.array([1, 0])
    elif name ==objectName[1]:
        tag = np.array([0,1])
    return tag

# combine labels and trainset
def trainset_with_label(path):

    images = []
    for name in os.listdir(path):
        object_path = os.path.join(path,name)
        for filename in tqdm(os.listdir(object_path)):
            full_path = os.path.join(object_path,filename)
            if full_path.endswith('png') or full_path.endswith('.jpg'):
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                images.append([np.array(img),label(str(name))])

    shuffle(images)
    return images

# combine labels and testset
def testset_with_label(path):

    images = []
    for name in os.listdir(path):
        object_path = os.path.join(path, name)
        for filename in tqdm(os.listdir(object_path)):
            full_path = os.path.join(object_path, filename)
            if full_path.endswith('png') or full_path.endswith('.jpg'):
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                images.append([np.array(img), label(str(name))])
    shuffle(images)
    return images


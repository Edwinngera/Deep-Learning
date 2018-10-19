import cv2
import os
from random import shuffle
from tqdm import tqdm
import  numpy as np
import PIL

oldPathName = '/Users/wang/Desktop/MiniProject/oldPathName/'
newPathName ='/Users/wang/Desktop/MiniProject/Dataset/'
objectName = ['cat','dog']
trainPathName = '/Users/wang/Desktop/MiniProject/cat-and-dog/Train/'
testPathName = '/Users/wang/Desktop/MiniProject/cat-and-dog/Test/'

def mkdir(newPathName):

    folder = os.path.exists(newPathName)

    if not folder:  # if the folder does not  exist, make it.
        os.makedirs(newPathName)
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There exists this folder!  ---")

def rename():
    mkdir(newPathName)
    i = 1
    for item in os.listdir(oldPathName):
        if item.endswith('.jpg') or item.endswith('.png'):
            old_name = oldPathName + str(item)
            new_name = newPathName+objectName[0]+'.'+str(i)+'.jpg'
            img = PIL.Image.open(old_name)
            img.save(new_name)
            i = i + 1

def label(img):
    lab = img.split('.')[0]
    if lab == objectName[0]:
        # cat is [1,0]
        tag = np.array([1,0])
    elif lab == objectName[1]:
        # dog is [0,1]
        tag = np.array([0,1])
    return tag

def train_data_with_label():
    train_images = []
    for fileName in tqdm(os.listdir(trainPathName)):
        path = os.path.join(trainPathName,fileName)
        if path.endswith('png') or path.endswith('.jpg'):
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
            train_images.append([np.array(img),label(fileName)])
    shuffle(train_images)
    return train_images

def test_data_with_label():
    test_images = []
    for fileName in tqdm(os.listdir(testPathName)):
        path = os.path.join(testPathName, fileName)
        if path.endswith('png')or path.endswith('.jpg'):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64),interpolation=cv2.INTER_CUBIC)
            test_images.append([np.array(img), label(fileName)])
    return test_images
import cv2
import os
from random import shuffle
from tqdm import tqdm
import  numpy as np



DataSetPathName ='/Users/wang/Desktop/MiniProject/Dataset/'
# Store the object name you want to classify
objectName = ['cat','dog']
trainPathName = '/Users/wang/Desktop/MiniProject/cat-and-dog/Train/'
testPathName = '/Users/wang/Desktop/MiniProject/cat-and-dog/Test/'

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

# get the label from images name
def label(img):
    lab = img.split('.')[0]
    if lab == objectName[0]:
        # cat is [1,0]
        tag = np.array([1,0])
    elif lab == objectName[1]:
        # dog is [0,1]
        tag = np.array([0,1])
    return tag

# resize and shuffle the train images
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

# resize the test images
def test_data_with_label():
    test_images = []
    for fileName in tqdm(os.listdir(testPathName)):
        path = os.path.join(testPathName, fileName)
        if path.endswith('png')or path.endswith('.jpg'):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64),interpolation=cv2.INTER_CUBIC)
            test_images.append([np.array(img), label(fileName)])
    return test_images
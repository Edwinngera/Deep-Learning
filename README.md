# Deep-Learning
This project is an image classification based on keras and tensorflow
The image resolution is set to be 64*64.
## CNN model -- image classification of five classes of objects or more
    file:
    CNN_Image_Classification.py
### Precondition 
#### Dataset
    (1) Download from kaggle (5 classes of flowers)    
[Lets go to Kaggle]( https://www.kaggle.com/alxmamaev/flowers-recognition)      
    
    (2) For the dataset, the file name is the label name
### Operation instruction
    (1) run the CNN_Image_Classifition.py
    (2) There will be a graph about accuracy and loss
### Results
##### learning rate = 1e-4
![Image](https://github.com/zywan/Deep-Learning/blob/master/lr%3D10%20-4.png)
##### learning rate = 5e-5
![Image](https://github.com/zywan/Deep-Learning/blob/master/lr%20%3D%205%2010%20-5.png)
##### learning rate = 4e-5
![Image](https://github.com/zywan/Deep-Learning/blob/master/new_acc.png)
##### learning rate = 2e-5
![Image](https://github.com/zywan/Deep-Learning/blob/master/lr%20%3D%202%2010%20-%205.png)

## Keras -- image classification of two classes of objects
    file list:
    (1) DataProcess.py
    (2) model.py
    (3) main.py
### Precondition
#### 1. install neccesary packages
    (1) numpy   
        pip3 install numpy
    (2) matplotlib
        python3 -m pip install -U matplotlib
    (3) keras
        pip3 install keras
    (4) tqdm
        pip3 install tqdm
    (5) cv2
        pip3 install opencv-python
#### 2. prepare dataset
    (1) Download from kaggle ( 2 classs, such as dog and cat)
[lets go to kaggle](https://www.kaggle.com/rahul897/catsdogs)     

    (2) google 
        create several folders and store images in them, folder name will be the label name for the images in it.
    (3) For the dataset, you have to have two folder, one is trainset, other is testset.     
        30% of the trainset will be validation set
    (4) For this program, you can only classify two class of object

### Operation instruction
#### 1. basic usage
    (1) After preparing the dataset and basic path name modification, you can run the main.py.    
    (2) Using checkpoint to store the best model every epoch and the program will load it after.
    (3) There will be a brief model summray. After that, you can see some test example with images and labels
        and also graphs about accuracy and loss.
#### 2. API
    import the python file. There are many function that you can use as well as the model.
### Results
#### Accuracy
![Image](https://github.com/zywan/Deep-Learning/blob/master/Accuracy.png)
#### Loss
![Image](https://github.com/zywan/Deep-Learning/blob/master/loss.png)
#### Some examples
![Image](https://github.com/zywan/Deep-Learning/blob/master/Figure_1.png)


## Comparison of two classification model
### model achitecture
Basically, the main difference between the two models are the convolutional layers, one have 3 convolutional layer and another have two, 
I tried to use 5 or more convolutional layers but it does not improve the performance a lot, and for mac, there is no NVIDIA GPU, the process would last for so long if the model is complicated, so I just simplify the model and guarantee a certain level of accuracy.
### Performance
As you can see from the graph above, both of the model have some overfitting issues, as for the five classes image classification, accuracy of 70% is fine, but for the 2 classes classification, the validation set accuracy is not that high may be because of the resolution of the images and also the learning rate, drop out and some other elements.          
I change the learning rate to restrain some overfitting but the result is not that good, but it is in a reasonable range.


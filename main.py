import  numpy as np
import DataProcess
import os
import model as m
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


def main():
    FolderRequest = str(input('Do you have data set? Y for yes and N for no:'))

    if FolderRequest == 'N':
        DataProcess.mkdir(DataProcess.DataSetPathName)
        print('Now, you can store your dataset in this new folder,the program is end now,'
              'please prepare your dataset and run the program again')
        os._exit(0)

    RenameRequest = str(input('Do you rename the data set to right formula? Y for yes and N for no:'))

    if RenameRequest == 'N':
        print('Now, you should rename the image,the program is end now,'
              'please finish the rename process run the program again')
        os._exit(0)


    training_images = DataProcess.train_data_with_label()
    testing_images = DataProcess.test_data_with_label()
    train_img_data = np.array([i[0] for i in training_images]).reshape(-1, 64, 64, 1)
    train_lbl_data = np.array([i[1] for i in training_images])
    test_img_data = np.array([i[0] for i in testing_images]).reshape(-1, 64, 64, 1)
    test_lbl_data = np.array([i[1] for i in testing_images])

    # load model
    if os.path.isfile("weights.best.hdf5"):
        m.model.load_weights("weights.best.hdf5")

    m.model.compile(optimizer=m.optimizer,loss = 'categorical_crossentropy',metrics=['accuracy'])

    # checkpoint to store the best model
    filepath = './weights.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # 30% of the train image will be the validation set
    history = m.model.fit(x=train_img_data,y=train_lbl_data,epochs=30,batch_size=50,validation_split=0.3,callbacks=callbacks_list)
    m.model.summary()

    # Plot the curve graph of loss and accuracy
    # Plot the training & validation accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot some images and labels from test set
    fig = plt.figure(figsize=(40,40))
    for i,data in enumerate(testing_images[20:50]):
        y = fig.add_subplot(6,5,i+1)
        img = data[0]
        data = img.reshape(1, 64,64, 1)
        model_out = m.model.predict([data])

        if np.argmax(model_out)==1:
            str_label = DataProcess.objectName[0]
        else:
            str_label = DataProcess.objectName[1]

        y.imshow(img,cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    main()
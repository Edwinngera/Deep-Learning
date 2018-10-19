import  numpy as np
import DataProcess
import os
import model as m
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
request = str(input('Do you have data set? Y for yes and N for no:'))

if request == 'N':
    DataProcess.rename()

training_images = DataProcess.train_data_with_label()
testing_images = DataProcess.test_data_with_label()
train_img_data = np.array([i[0] for i in training_images]).reshape(-1, 64, 64, 1)
train_lbl_data = np.array([i[1] for i in training_images])
test_img_data = np.array([i[0] for i in testing_images]).reshape(-1, 64, 64, 1)
test_lbl_data = np.array([i[1] for i in testing_images])

# load weights
if os.path.isfile("weights.best.hdf5"):
    m.model.load_weights("weights.best.hdf5")

m.model.compile(optimizer=m.optimizer,loss = 'categorical_crossentropy',metrics=['accuracy'])
# checkpoint
#filepath="./weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath = './weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = m.model.fit(x=train_img_data,y=train_lbl_data,epochs=30,batch_size=50,validation_split=0.3,callbacks=callbacks_list)
m.model.summary()

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
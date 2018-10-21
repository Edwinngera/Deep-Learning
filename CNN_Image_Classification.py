import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from datetime import timedelta
import random
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle

# Function of loading dataset
def load_train(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('Reading training images')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), interpolation = cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls

def load_test(test_path, image_size):
    path = os.path.join(test_path, '*g')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    print("Reading test images")
    for fl in files:
        flbase = os.path.basename(fl)
        img = cv2.imread(fl)
        img = cv2.resize(img, (image_size, image_size), interpolation = cv2.INTER_LINEAR)
        X_test.append(img)
        X_test_id.append(flbase)

### because we're not creating a DataSet object for the test images, normalization happens here
    X_test = np.array(X_test, dtype=np.uint8)
    X_test = X_test.astype('float32')
    X_test = X_test / 255

    return X_test, X_test_id

class DataSet(object):

    def __init__(self, images, labels, ids, cls):
        self._num_examples = images.shape[0]
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    def images(self):
        return self._images

    def labels(self):
        return self._labels

    def ids(self):
        return self._ids

    def cls(self):
        return self._cls

    def num_examples(self):
        return self._num_examples

    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size=0):
    class DataSets(object):
        pass
    data_sets = DataSets()

    images, labels, ids, cls = load_train(train_path, image_size, classes)
    images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]
        validation_ids = ids[:validation_size]
        validation_cls = cls[:validation_size]

        train_images = images[validation_size:]
        train_labels = labels[validation_size:]
        train_ids = ids[validation_size:]
        train_cls = cls[validation_size:]

        data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
        data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

    return data_sets


def read_test_set(test_path, image_size):
    images, ids  = load_test(test_path, image_size)
    return images, ids

# Convolutional Layer 1.
filter_size1 = 5 
num_filters1 = 64

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 64

# Fully-connected layer 1.
fc1_size = 128

# Fully-connected layer 2.
fc2_size = 128

# Number of color channels
num_channels = 3

# image dimensions
img_size = 64

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

img_shape = (img_size, img_size)

classes = ['daisy','dandelion','rose','sunflower','tulip']

num_classes = len(classes)

# batch size
batch_size = 32

# validation split
validation_size = 0.3


train_path = 'flowers'

checkpoint_dir = "ckpoint"

# load training dataset
data = read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels())))
print("- Validation:\t{}".format(len(data.valid.labels())))


### Helper-function for plotting images
def plot_images(images, cls_true, cls_pred=None):
    
    if len(images()) == 0:
        print("no images to show")
        return 
    else:
        random_indices = random.sample(range(len(images())), min(len(images()), 9))
  
    images, cls_true  = zip(*[(images()[i], cls_true()[i]) for i in random_indices])
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_size, img_size, num_channels))

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "label: {0}".format(cls_true[i])
        else:
            xlabel = "label: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()

# Get some random images and their labels from the train set.
images, cls_true  = data.train.images, data.train.cls
plot_images(images=images, cls_true=cls_true)


#TensorFlow Graph
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
# Convolutional Layer
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    layer += biases

    # Use pooling to down-sample the image resolution
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    return layer, weights


#  Flattening a layer
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]
    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

# Fully-Connected Layer
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# ### Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


# Convolutional Layer 1
layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

# Convolutional Layers 2 】
layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)


# Flatten Layer

layer_flat, num_features = flatten_layer(layer_conv2)
print(layer_flat, num_features)

# Fully-Connected Layer 1
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc1_size,
                         use_relu=True)

# Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc1_size,
                         num_outputs=num_classes,
                         use_relu=False)
# Predicted Class
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# Cost-function to be optimized
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=4e-5).minimize(cost)

# Performance Measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# TensorFlow Run
session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = batch_size
global acc_matrix
acc_matrix = []
global val_acc_matrix
val_acc_matrix = []
global val_loss_matrix
val_loss_matrix= []
def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))
    acc_matrix.append(acc)
    val_acc_matrix.append(val_acc)
    val_loss_matrix.append(val_loss)
# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    epoch_range= []
    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]

        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,y_true: y_true_batch}
        
        feed_dict_validate = {x: x_valid_batch,y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples()/batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples()/batch_size))
            epoch_range.append(epoch+1)
            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(epoch_range,acc_matrix,label = 'Training acc')
    plt.plot(epoch_range,val_acc_matrix,label = 'Validation acc')
    plt.legend(loc = 'upper right')
    plt.subplot(1,2,2)
    plt.plot(epoch_range,val_loss_matrix, label = 'Validation loss')
    plt.legend(loc = 'upper right')
    plt.show()

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))
optimize(num_iterations=5000)

#x_test = data.valid.images().reshape(399, img_size_flat)
x_test = data.valid.images().reshape(1296, img_size_flat)
feed_dict_test = {x: x_test, y_true: data.valid.labels()}
val_loss = session.run(cost, feed_dict=feed_dict_test)
val_acc = session.run(accuracy, feed_dict=feed_dict_test)

msg_test = "Test Accuracy: {0:>6.1%}"
print(msg_test.format(val_acc))
########################################################################
#
# Cache-wrapper for a function or class.
#
# Save the result of calling a function or creating an object-instance
# to harddisk. This is used to persist the data so it can be reloaded
# very quickly and easily.
#
# Implemented in Python 3.5
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import os
import pickle
import numpy as np

########################################################################


def cache(cache_path, fn, *args, **kwargs):
    """
    Cache-wrapper for a function or class. If the cache-file exists
    then the data is reloaded and returned, otherwise the function
    is called and the result is saved to cache. The fn-argument can
    also be a class instead, in which case an object-instance is
    created and saved to the cache-file.

    :param cache_path:
        File-path for the cache-file.

    :param fn:
        Function or class to be called.

    :param args:
        Arguments to the function or class-init.

    :param kwargs:
        Keyword arguments to the function or class-init.

    :return:
        The result of calling the function or creating the object-instance.
    """

    # If the cache-file exists.
    if os.path.exists(cache_path):
        # Load the cached data from the file.
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)

        print("- Data loaded from cache-file: " + cache_path)
    else:
        # The cache-file does not exist.

        # Call the function / class-init with the supplied arguments.
        obj = fn(*args, **kwargs)

        # Save the data to a cache-file.
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)

        print("- Data saved to cache-file: " + cache_path)

    return obj


########################################################################


def convert_numpy2pickle(in_path, out_path):
    """
    Convert a numpy-file to pickle-file.

    The first version of the cache-function used numpy for saving the data.
    Instead of re-calculating all the data, you can just convert the
    cache-file using this function.

    :param in_path:
        Input file in numpy-format written using numpy.save().

    :param out_path:
        Output file written as a pickle-file.

    :return:
        Nothing.
    """

    # Load the data using numpy.
    data = np.load(in_path)

    # Save the data using pickle.
    with open(out_path, mode='wb') as file:
        pickle.dump(data, file)


########################################################################

if __name__ == '__main__':
    # This is a short example of using a cache-file.

    # This is the function that will only get called if the result
    # is not already saved in the cache-file. This would normally
    # be a function that takes a long time to compute, or if you
    # need persistent data for some other reason.
    def expensive_function(a, b):
        return a * b

    print('Computing expensive_function() ...')

    # Either load the result from a cache-file if it already exists,
    # otherwise calculate expensive_function(a=123, b=456) and
    # save the result to the cache-file for next time.
    result = cache(cache_path='cache_expensive_function.pkl',
                   fn=expensive_function, a=123, b=456)

    print('result =', result)

    # Newline.
    print()

    # This is another example which saves an object to a cache-file.

    # We want to cache an object-instance of this class.
    # The motivation is to do an expensive computation only once,
    # or if we need to persist the data for some other reason.
    class ExpensiveClass:
        def __init__(self, c, d):
            self.c = c
            self.d = d
            self.result = c * d

        def print_result(self):
            print('c =', self.c)
            print('d =', self.d)
            print('result = c * d =', self.result)

    print('Creating object from ExpensiveClass() ...')

    # Either load the object from a cache-file if it already exists,
    # otherwise make an object-instance ExpensiveClass(c=123, d=456)
    # and save the object to the cache-file for the next time.
    obj = cache(cache_path='cache_ExpensiveClass.pkl',
                fn=ExpensiveClass, c=123, d=456)

    obj.print_result()

########################################################################
########################################################################
#
# Functions for downloading and extracting data-files from the internet.
#
# Implemented in Python 3.5
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import sys
import os
import urllib.request
import tarfile
import zipfile

########################################################################


def _print_download_progress(count, block_size, total_size):
    """
    Function used for printing the download progress.
    Used as a call-back function in maybe_download_and_extract().
    """

    # Percentage completion.
    pct_complete = float(count * block_size) / total_size

    # Status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


########################################################################


def maybe_download_and_extract_of_download(url, download_dir):
    """
    Download and extract the data if it doesn't already exist.
    Assumes the url is a tar-ball file.

    :param url:
        Internet URL for the tar-file to download.
        Example: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    :param download_dir:
        Directory where the downloaded file is saved.
        Example: "data/CIFAR-10/"

    :return:
        Nothing.
    """

    # Filename for saving the file downloaded from the internet.
    # Use the filename from the URL and add it to the download_dir.
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download the file from the internet.
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")


########################################################################
########################################################################
#
# Class for creating a data-set consisting of all files in a directory.
#
# Example usage is shown in the file knifey.py and Tutorial #09.
#
# Implemented in Python 3.5
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import os

########################################################################


def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.

    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]

    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.

    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.

    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]


########################################################################


class DataSet:
    def __init__(self, in_dir, exts='.jpg'):
        """
        Create a data-set consisting of the filenames in the given directory
        and sub-dirs that match the given filename-extensions.

        For example, the knifey-spoony data-set (see knifey.py) has the
        following dir-structure:

        knifey-spoony/forky/
        knifey-spoony/knifey/
        knifey-spoony/spoony/
        knifey-spoony/forky/test/
        knifey-spoony/knifey/test/
        knifey-spoony/spoony/test/

        This means there are 3 classes called: forky, knifey, and spoony.

        If we set in_dir = "knifey-spoony/" and create a new DataSet-object
        then it will scan through these directories and create a training-set
        and test-set for each of these classes.

        The training-set will contain a list of all the *.jpg filenames
        in the following directories:

        knifey-spoony/forky/
        knifey-spoony/knifey/
        knifey-spoony/spoony/

        The test-set will contain a list of all the *.jpg filenames
        in the following directories:

        knifey-spoony/forky/test/
        knifey-spoony/knifey/test/
        knifey-spoony/spoony/test/

        See the TensorFlow Tutorial #09 for a usage example.

        :param in_dir:
            Root-dir for the files in the data-set.
            This would be 'knifey-spoony/' in the example above.

        :param exts:
            String or tuple of strings with valid filename-extensions.
            Not case-sensitive.

        :return:
            Object instance.
        """

        # Extend the input directory to the full path.
        in_dir = os.path.abspath(in_dir)

        # Input directory.
        self.in_dir = in_dir

        # Convert all file-extensions to lower-case.
        self.exts = tuple(ext.lower() for ext in exts)

        # Names for the classes.
        self.class_names = []

        # Filenames for all the files in the training-set.
        self.filenames = []

        # Filenames for all the files in the test-set.
        self.filenames_test = []

        # Class-number for each file in the training-set.
        self.class_numbers = []

        # Class-number for each file in the test-set.
        self.class_numbers_test = []

        # Total number of classes in the data-set.
        self.num_classes = 0

        # For all files/dirs in the input directory.
        for name in os.listdir(in_dir):
            # Full path for the file / dir.
            current_dir = os.path.join(in_dir, name)

            # If it is a directory.
            if os.path.isdir(current_dir):
                # Add the dir-name to the list of class-names.
                self.class_names.append(name)

                # Training-set.

                # Get all the valid filenames in the dir (not sub-dirs).
                filenames = self._get_filenames(current_dir)

                # Append them to the list of all filenames for the training-set.
                self.filenames.extend(filenames)

                # The class-number for this class.
                class_number = self.num_classes

                # Create an array of class-numbers.
                class_numbers = [class_number] * len(filenames)

                # Append them to the list of all class-numbers for the training-set.
                self.class_numbers.extend(class_numbers)

                # Test-set.

                # Get all the valid filenames in the sub-dir named 'test'.
                filenames_test = self._get_filenames(os.path.join(current_dir, 'test'))

                # Append them to the list of all filenames for the test-set.
                self.filenames_test.extend(filenames_test)

                # Create an array of class-numbers.
                class_numbers = [class_number] * len(filenames_test)

                # Append them to the list of all class-numbers for the test-set.
                self.class_numbers_test.extend(class_numbers)

                # Increase the total number of classes in the data-set.
                self.num_classes += 1

    def _get_filenames(self, dir):
        """
        Create and return a list of filenames with matching extensions in the given directory.

        :param dir:
            Directory to scan for files. Sub-dirs are not scanned.

        :return:
            List of filenames. Only filenames. Does not include the directory.
        """

        # Initialize empty list.
        filenames = []

        # If the directory exists.
        if os.path.exists(dir):
            # Get all the filenames with matching extensions.
            for filename in os.listdir(dir):
                if filename.lower().endswith(self.exts):
                    filenames.append(filename)

        return filenames

    def get_paths(self, test=False):
        """
        Get the full paths for the files in the data-set.

        :param test:
            Boolean. Return the paths for the test-set (True) or training-set (False).

        :return:
            Iterator with strings for the path-names.
        """

        if test:
            # Use the filenames and class-numbers for the test-set.
            filenames = self.filenames_test
            class_numbers = self.class_numbers_test

            # Sub-dir for test-set.
            test_dir = "test/"
        else:
            # Use the filenames and class-numbers for the training-set.
            filenames = self.filenames
            class_numbers = self.class_numbers

            # Don't use a sub-dir for test-set.
            test_dir = ""

        for filename, cls in zip(filenames, class_numbers):
            # Full path-name for the file.
            path = os.path.join(self.in_dir, self.class_names[cls], test_dir, filename)

            yield path

    def get_training_set(self):
        """
        Return the list of paths for the files in the training-set,
        and the list of class-numbers as integers,
        and the class-numbers as one-hot encoded arrays.
        """

        return list(self.get_paths()), \
               np.asarray(self.class_numbers), \
               one_hot_encoded(class_numbers=self.class_numbers,
                               num_classes=self.num_classes)

    def get_test_set(self):
        """
        Return the list of paths for the files in the test-set,
        and the list of class-numbers as integers,
        and the class-numbers as one-hot encoded arrays.
        """

        return list(self.get_paths(test=True)), \
               np.asarray(self.class_numbers_test), \
               one_hot_encoded(class_numbers=self.class_numbers_test,
                               num_classes=self.num_classes)


########################################################################


def load_cached(cache_path, in_dir):
    """
    Wrapper-function for creating a DataSet-object, which will be
    loaded from a cache-file if it already exists, otherwise a new
    object will be created and saved to the cache-file.

    This is useful if you need to ensure the ordering of the
    filenames is consistent every time you load the data-set,
    for example if you use the DataSet-object in combination
    with Transfer Values saved to another cache-file, see e.g.
    Tutorial #09 for an example of this.

    :param cache_path:
        File-path for the cache-file.

    :param in_dir:
        Root-dir for the files in the data-set.
        This is an argument for the DataSet-init function.

    :return:
        The DataSet-object.
    """

    print("Creating dataset from the files in: " + in_dir)

    # If the object-instance for DataSet(in_dir=data_dir) already
    # exists in the cache-file then reload it, otherwise create
    # an object instance and save it to the cache-file for next time.
    dataset = cache(cache_path=cache_path,
                    fn=DataSet, in_dir=in_dir)

    return dataset


########################################################################
########################################################################
#
# Functions for downloading the CIFAR-10 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import pickle
import os

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "data/CIFAR-10/"

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

########################################################################
# Private functions for downloading, unpacking and loading data-files.


def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.

    If filename=="" then return the directory of the files.
    """

    return os.path.join(data_path, "cifar-10-batches-py/", filename)


def _unpickle(filename):
    """
    Unpickle the given file and return the data.

    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file,  encoding='latin1')

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data['data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data['labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def maybe_download_and_extract():
    """
    Download and extract the CIFAR-10 data-set if it doesn't already exist
    in data_path (set this variable first to the desired path).
    """

    maybe_download_and_extract_of_download(url=data_url, download_dir=data_path)


def load_class_names():
    """
    Load the names for the classes in the CIFAR-10 data-set.

    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """

    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")['label_names']

    # Convert from binary strings.
    return raw


def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.

    The data-set is split into 5 data-files which are merged here.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(filename="test_batch")

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

########################################################################

maybe_download_and_extract()
class_names = load_class_names()
images_train, cls_train, labels_train = load_training_data()
images_test, cls_test, labels_test = load_test_data()

from sklearn import ensemble
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
# from google.colab import files
import csv
from sklearn.model_selection import cross_val_score

print(type(images_train), type(cls_train), type(labels_train))
print(images_train.shape, cls_train.shape, labels_train.shape)
# print(images_train[0][0])
# for i in range (len(labels_train)):
#     print(labels_train[i])
def PCA_implementation(images_train, images_test):
    # fig = plt.figure(figsize=(10, 10))
    # for i in range(100):
    #     ax = fig.add_subplot(10, 10, i + 1)
    #     ax.imshow(images_train[i], cmap=plt.cm.bone)
    # plt.show()
    x = images_train.copy()
    x_test = images_test.copy()
    x = x.reshape((50000, 32 * 32 * 3))
    x_test = x_test.reshape((10000, 32 * 32 * 3))
    x = preprocessing.scale(x)
    x_test = preprocessing.scale(x_test)
    pca1 = PCA()
    pca1.fit(x)
    k = 0
    current_sum = 0
    total = sum(pca1.explained_variance_)
    while current_sum / total < 0.99:
        current_sum += pca1.explained_variance_[k]
        k = k + 1
    pca = PCA(n_components=k, whiten=True)
    x_transform_pca = pca.fit_transform(x)
    x_test_pca=pca.transform(x_test)
    x_approx = pca.inverse_transform(x_transform_pca)
#     x_approx.reshape((50000, 32, 32, 3))
#     fig = plt.figure(figsize=(10, 10))
#     for i in range(100):
#         ax = fig.add_subplot(10, 10, i + 1)
#         ax.imshow(x_approx[i], cmap=plt.cm.bone)
#     plt.show()
    return x_transform_pca,x_test_pca

x_train,x_test=PCA_implementation(images_train,images_test)

def random_forest(x_train,x_test,y_train,y_test):
    rf=ensemble.RandomForestClassifier()
    rf.fit(x_train,y_train)
    y_pred=rf.predict(x_test)
    print("Random Forest.................................")
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    return y_pred

def logistic_regression(x_train,x_test,y_train,y_test):
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    print("Logistic Regression.................................")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return y_pred

def svm(x_train,x_test,y_train,y_test):
  clf = SVC(kernel = 'linear')
  clf.fit(x_train, y_train)
  y_pred = clf.predict(x_test)
  print("SVM.................................")
  print(classification_report(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))
  return y_pred

def KNN(x_train,x_test,y_train,y_test):
  x_axis = []
  y_axis = []
  for i in range(1, 26, 2):
    clf = KNeighborsClassifier(n_neighbors = i)
    score = cross_val_score(clf, x_train, y_train)
    x_axis.append(i)
    y_axis.append(score.mean())
  max_score=(max(y_axis))
  for i in range(len(y_axis)):
    if y_axis[i]==max_score:
        val=x_axis[i]
  neigh=KNeighborsClassifier(n_neighbors = val)
  neigh.fit(x_train, y_train)
  y_pred = neigh.predict(x_test)
  print("KNN.................................")
  print(classification_report(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))
  return y_pred

def Decision_Tree(x_train,x_test,y_train,y_test):
  clf=DecisionTreeClassifier()
  clf.fit(x_train,y_train)
  y_pred = clf.predict(x_test)
  print("Decision Tree............................")
  print(classification_report(y_test,y_pred))
  print(confusion_matrix(y_test,y_pred))
  return y_pred
y_train=cls_train.copy()
y_test=cls_test.copy()
labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

y_pred_1=random_forest(x_train,x_test,y_train,y_test)
y_final_1=[]
for i in y_pred_1:
  y_final_1.append([labels[i]])
with open('sub_1.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\n')
    for i in range(len(y_final_1)):
      spamwriter.writerow(y_final_1[i])
# files.download('sub_1.csv')


y_pred_2=logistic_regression(x_train,x_test,y_train,y_test)
y_final_2=[]
for i in y_pred_2:
  y_final_2.append([labels[i]])
with open('sub_2.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\n')
    for i in range(len(y_final_2)):
      spamwriter.writerow(y_final_2[i])
# files.download('sub_2.csv')


y_pred_3=KNN(x_train,x_test,y_train,y_test)
y_final_3=[]
for i in y_pred_3:
  y_final_3.append([labels[i]])
with open('sub_3.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\n')
    for i in range(len(y_final_3)):
      spamwriter.writerow(y_final_3[i])
# files.download('sub_3.csv')

y_pred_4=Decision_Tree(x_train,x_test,y_train,y_test)
y_final_4=[]
for i in y_pred_4:
  y_final_4.append([labels[i]])
with open('sub_4.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\n')
    for i in range(len(y_final_4)):
      spamwriter.writerow(y_final_4[i])
# files.download('sub_4.csv')


y_pred_5=svm(x_train,x_test,y_train,y_test)
y_final_5=[]
for i in y_pred_5:
  y_final_5.append([labels[i]])
with open('sub_5.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\n')
    for i in range(len(y_final_5)):
      spamwriter.writerow(y_final_5[i])
# files.download('sub_5.csv')

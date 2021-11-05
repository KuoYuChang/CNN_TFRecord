# CNN_TFRecord

## An example of Tensorflow pipeline with TFRecord
## Implement Tensorflow tutorial in Jetson Nano, tutorial from https://www.tensorflow.org/tutorials/images/cnn
### Since cpu and gpu shared memory in Jetson Nano, it's easily lead to out-of-memory when data and model both in memory. Te preserve memory for model, preprocess data with TFRecord, read a small batch of the dataset into memory.
### Using tensorflow 2.1.0
### CIFAR10 dataset, CNN classification
### Write images into TFRecord, load and establish pipeline with tf.data, without loading whole dataset
### Training and validating as the Tensorflow tutorial


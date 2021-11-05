import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from functools import partial

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, label):
  #image_shape = tf.image.decode_jpeg(image_string).shape
  
  feature = {
      #'height': _int64_feature(image_shape[0]),
      #'width': _int64_feature(image_shape[1]),
      #'depth': _int64_feature(image_shape[2]),
      'image_raw': _bytes_feature(image_string),
      'label': _int64_feature(label),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))
  
def wirte_record(record_file, num, images, labels):
    with tf.io.TFRecordWriter(record_file) as writer:
        for i in range(num):
            image = images[i]
            label = labels[i]

            #encode
            image_string = tf.image.encode_jpeg(image)

            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())
            del image_string
            del tf_example

            
# Create a dictionary describing the features.
image_feature_description = {
    #'height': tf.io.FixedLenFeature([], tf.int64),
    #'width': tf.io.FixedLenFeature([], tf.int64),
    #'depth': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_image_function(example_proto, image_shape, num_classes):
    # Parse the input tf.train.Example proto using the dictionary above.
    # do some preprocessing
    
    #image_shape = [32, 32, 3]
    #print(image_shape)
    
    # problems
    example = tf.io.parse_single_example(example_proto, image_feature_description)
    
    
    image = tf.image.decode_jpeg(example['image_raw'])
    image = tf.reshape(image, image_shape)
    # rescaling to [0,1]
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    
    label = tf.cast(example['label'], tf.uint8)
    #label = tf.one_hot(label, num_classes)
    #print(image)
    #print(label)
    
    #del example
    
    return image, label

'''
def set_pipeline(dataset, BATCH_SIZE):
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    #dataset = dataset.shuffle(2048)
    return dataset
'''

# modified to return 'object'
def get_dataset(filename, image_shape, num_classes, AUTOTUNE, BATCH_SIZE=32):
    dataset = tf.data.TFRecordDataset(filename)
    
    dataset = dataset.map( partial(_parse_image_function, image_shape=image_shape, num_classes=num_classes) )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.shuffle(128, reshuffle_each_iteration=True)
    
    # Create an iterator
    #iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    # Create your tf representation of the iterator
    #image, label = iterator.get_next()
    
    #image, label = next(iter(dataset))

    #return image, label
    return dataset

def show_batch(dataset):
    image_batch, label_batch = next(iter(dataset))
    for n in range(32):
        print(n)
        #print(image_batch[n])
        print(label_batch[n])
        plt.imshow(image_batch[n])
        plt.show()

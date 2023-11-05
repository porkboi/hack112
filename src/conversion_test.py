import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import tensorflow as tf
from PIL import Image

# Source: https://stackoverflow.com/questions/33849617/how-do-i-convert-a-directory-of-jpeg-images-to-tfrecords-file-in-tensorflow
# Note: modified from source
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# images and labels array as input
def convert_to(images, labels, output_directory, name):
    num_examples = len(labels)
    if len(images) != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (len(images), num_examples))
    #print(images)
    rows = len(images)

    filename = os.path.join(output_directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'image_2': _bytes_feature(image_raw),
            'label': _int64_feature(int(labels[index][6:]))}))
        writer.write(example.SerializeToString())

def read_image(file_name, images_path):
    image = Image.open(images_path + file_name)
    return image

def extract_image_index_make_label(img_name):
    label = img_name.split(".")[0]
    return label

images_path = "face_detection_model/"
image_list = os.listdir(images_path)
images = []
labels = []
for img_name in image_list:
    images.append(read_image(img_name, images_path))
    labels.append(extract_image_index_make_label(img_name))
#print(images_array)

convert_to(images, labels, ".", "test")
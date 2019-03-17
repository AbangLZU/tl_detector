#!/usr/bin/env python

import os
import csv
import io
import itertools
import hashlib
import random

import PIL.Image

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import dataset_util

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

ANNOTATION = 'frameAnnotationsBOX.csv'
FRAMES = 'frames'

MAP = {
    'go': 'green',
    'goLeft': 'green',
    'stop': 'red',
    'stopLeft': 'red',
    'warning': 'yellow',
    'warningLeft': 'yellow'
}


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to LISA dataset.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/lisa_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

width = None
height = None

def process_frame(label_map_dict, frame):
    global width
    global height

    filename, xmin, ymin, xmax, ymax, classes = frame

    if not os.path.exists(filename):
        tf.logging.error("File %s not found", filename)
        return

    with tf.gfile.GFile(filename, 'rb') as img:
        encoded_png = img.read()

    png = PIL.Image.open(io.BytesIO(encoded_png))
    if png.format != 'PNG':
        tf.logging.error("File %s has unexpeted image format '%s'", filename, png.format)
        return

    if width is None and height is None:
        width = png.width
        height = png.height
        tf.logging.info('Expected image size: %dx%d', width, height)

    if width != png.width or height != png.height:
        tf.logging.error('File %s has unexpected size', filename)
        return
    print filename
    print classes
    key = hashlib.sha256(encoded_png).hexdigest()
    labels = [ label_map_dict[c] for c in classes ]
    xmin = [ float(x)/width for x in xmin ]
    xmax = [ float(x)/width for x in xmax ]
    ymin = [ float(y)/height for y in ymin ]
    ymax = [ float(y)/height for y in ymax ]
    
    classes = [ c.encode('utf8') for c in classes ]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes),
        'image/object/class/label': dataset_util.int64_list_feature(labels)
    }))
    return example

def create_frame(root, frame, records):
    filename = os.path.join(root, FRAMES, os.path.basename(frame))
    if not os.path.exists(filename):
        tf.logging.error("File %s not found", filename)
        return

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []

    for r in records:
        if r['Annotation tag'] not in MAP:
            continue
        classes.append(MAP[r['Annotation tag']])
        xmin.append(r['Upper left corner X'])
        xmax.append(r['Lower right corner X'])
        ymin.append(r['Upper left corner Y'])
        ymax.append(r['Lower right corner Y'])

    yield (filename, xmin, ymin, xmax, ymax, classes)

def process_annotation(root):
    tf.logging.info('Processing %s', os.path.join(root, ANNOTATION))
    with tf.gfile.GFile(os.path.join(root, ANNOTATION)) as a:
        annotation = a.read().decode('utf-8')

    with io.StringIO(annotation) as a:
        data = csv.DictReader(a, delimiter=';')
        for key, group in itertools.groupby(data, lambda r: r['Filename']):
            for e in create_frame(root, key, group):
                yield e

def main(_):
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    frames = []
    for r, d, f in tf.gfile.Walk(FLAGS.data_dir, in_order=True):
        if ANNOTATION in f:
            del d[:]
            for e in process_annotation(r):
                frames.append(e)
    # random.shuffle(frames)

    num_shards=30

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, FLAGS.output_path, num_shards)

        for index, f in enumerate(frames):
            tf_example = process_frame(label_map_dict, f)

            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()







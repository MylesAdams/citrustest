import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import string

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sqlalchemy import create_engine
import json

app = Flask(__name__)
api = Api(app)

from google.appengine.ext import vendor

vendor.add('lib')


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError(
        'Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(
    '/usr/local/lib/python3.6/site-packages/tensorflow/models/research/object_detection/data/', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

counter = 0


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def analyze_image(filepath):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as session:
            image = Image.open(filepath)
            image_np = load_image_into_numpy_array(image)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = session.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            height, width, _ = image_np.shape

            final_score = np.squeeze(scores)
            count = 0
            for i in range(100):
                if scores is None or final_score[i] > 0.5:
                    count = count + 1
            print("num_detections: " + str(count))

            objects = []

            for i in range(0, count):
                data = {}

                score = scores[0][i]

                if (score > .65):
                    y1, x1, y2, x2 = boxes[0][i]
                    y1_o = int(y1 * height)
                    x1_o = int(x1 * width)
                    y2_o = int(y2 * height)
                    x2_o = int(x2 * width)
                    predicted_class = category_index[classes[0][i]]['name']

                    box_width = x2_o - x1_o
                    box_height = y2_o - y1_o

                    x_pad = .1 * box_width
                    if (x1_o - x_pad > 0):
                        x1_o -= x_pad
                    else:
                        x1_o = 0

                    if (x2_o + x_pad < width):
                        x2_o += x_pad
                    else:
                        x2_o = width - 1

                    y_pad = .1 * box_height
                    if (y1_o - y_pad > 0):
                        y1_o -= y_pad
                    else:
                        y1_o = 0

                    if (y2_o + y_pad < height):
                        y2_o += y_pad
                    else:
                        y2_o = height - 1

                    data['score'] = score
                    data['boundbox'] = [int(x1_o), int(
                        y1_o), int(x2_o), int(y2_o)]
                    data['class'] = predicted_class

                    objects.append(data)

                    print(
                        "score: " + str(score) +
                        "\nbb_o: " + str([x1_o, y1_o, x2_o, y2_o]) +
                        "\nimg_size: " + str([height, width]) +
                        "\nclass: " + str(predicted_class + "\n"))

            objects_json = jsonify(str(objects))

            print(objects_json)

            return objects_json


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        req = request.headers

        file = req['filename'] + '.' + req['filetype']

        with open(file, 'wb') as out:
            out.write(request.data)

        json = analyze_image(file)

        return json


if __name__ == '__main__':
    app.run(debug=True, port=5000)  # run app in debug mode on port 5000

from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import numpy as np
import cv2
from PIL import Image
from tensorflow.python.saved_model import tag_constants
from core.functions import *
from core.yolov4 import filter_boxes
import core.utils as utils
from absl.flags import FLAGS
from absl import app, flags, logging
import tensorflow as tf
import time
import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def showingResize(img, scale_percent=60):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    cv2.imshow('result', resized)


def capture():
    image = cv2.imread('api.jpg')
    return image


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(
    tiny=False, model='yolov4')
input_size = 416
model_path = './checkpoints/firefighter.tflite'

interpreter = tf.lite.Interpreter(model_path=model_path)
image_ori = capture()
image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)

image_data = cv2.resize(image_ori, (input_size, input_size))
image_data = image_data / 255.
images_data = image_data[np.newaxis, ...].astype(np.float32)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], images_data)
interpreter.invoke()
pred = [interpreter.get_tensor(output_details[i]['index'])
        for i in range(len(output_details))]

boxes, pred_conf = filter_boxes(
    pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))

boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
    scores=tf.reshape(
        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
    max_output_size_per_class=50,
    max_total_size=50,
    iou_threshold=0.45,
    score_threshold=0.5
)
original_h, original_w, _ = image_ori.shape
bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

# hold all detection data in one variable
pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0],
             valid_detections.numpy()[0]]

# read in all class names from config
class_names = utils.read_class_names(cfg.YOLO.CLASSES)

# by default allow all classes in .names file
allowed_classes = list(class_names.values())
image = utils.draw_bbox(
    image_ori,
    pred_bbox,
    allowed_classes=allowed_classes
)
image = Image.fromarray(image.astype(np.uint8))
image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
cv2.imwrite('detections.png', image)

# showingResize(image, 40)
# cv2.waitKey(0)

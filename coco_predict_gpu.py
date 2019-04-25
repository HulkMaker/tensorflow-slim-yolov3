#! /usr/bin/env python
# coding=utf-8


import sys
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from core import utils, yolov3
from core.dataset import dataset, Parser

coco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
            89, 90]

sess = tf.Session()


IMAGE_H, IMAGE_W = 416, 416
CLASSES          = utils.read_coco_names('./data/coco.names')
NUM_CLASSES      = len(CLASSES)
ANCHORS          = utils.get_anchors('./data/coco_anchors.txt', IMAGE_H, IMAGE_W)
CKPT_FILE        = "/home/common/pretrained_models/checkpoint/yolov3.ckpt"
IOU_THRESH       = 0.5
SCORE_THRESH     = 0.001

all_detections   = []
all_annotations  = []
all_aver_precs   = {CLASSES[i]:0. for i in range(NUM_CLASSES)}

test_tfrecord    = "/home/common/datasets/tfrecords/5k.tfrecords"
parser           = Parser(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
testset          = dataset(parser, test_tfrecord , batch_size=1, shuffle=None, repeat=False)


images_tensor, *y_true_tensor  = testset.get_next()
model = yolov3.yolov3(NUM_CLASSES, ANCHORS)
with tf.variable_scope('yolov3'):
    pred_feature_map    = model.forward(images_tensor, is_training=False)
    y_pred_tensor       = model.predict(pred_feature_map)

saver = tf.train.Saver()
saver = tf.train.import_meta_graph('/home/common/pretrained_models/checkpoint/yolov3.ckpt.meta')
saver.restore(sess, CKPT_FILE)
dt_result_path = "results/pt_cocoapi.json"
imglist_path = "data/4954.txt"
image_bbox_path="bbox/"
# total_img_num=5000
if os.path.exists(dt_result_path):
    os.remove(dt_result_path)

labels = utils.read_coco_names('./data/coco.names')

with open(imglist_path) as f:
    total_img_list = f.readlines()
    total_img_list = [x.strip() for x in total_img_list]
    total_img_num = len(total_img_list)

with open(dt_result_path, "a") as new_p:
    image_idx = 0
    new_p.write("[")

    for image_path in total_img_list:

        if (os.path.exists(image_path)):
            print(image_idx, image_path)

        orig_index = int(image_path[50:56])
        img = Image.open(image_path)
        img = img.convert('RGB')
        orig_width, orig_height = img.size

        y_pred, y_true, image  = sess.run([y_pred_tensor, y_true_tensor, images_tensor])
        pred_boxes = y_pred[0][0]
        pred_confs = y_pred[1][0]
        pred_probs = y_pred[2][0]
        image      = Image.fromarray(np.uint8(image[0]*255))

        boxes, scores, classes = utils.cpu_nms(pred_boxes, pred_confs*pred_probs, NUM_CLASSES,
                                                      score_thresh=SCORE_THRESH, iou_thresh=IOU_THRESH)
        classes = [] if classes is None else classes.tolist()
        # print("pred_labels_list",pred_labels_list)

        # image_bbox = utils.draw_boxes(img, boxes, scores, classes, labels, [IMAGE_H, IMAGE_W], is_show=False)
        # image_bbox.save(image_bbox_path+str(orig_index)+'.jpg')

        for j in range(len(classes)):
            coco_id = coco_ids[int(classes[j])]
            left,top ,right,bottom = boxes[j]

            left = max(left, 0)
            top = max(top, 0)
            right = min(right, 416)
            bottom = min(bottom, 416)

            left = round(left * orig_width / 416, 4)
            top = round(top * orig_height / 416, 4)
            right = right * orig_width / 416
            bottom = bottom * orig_height / 416

            width = round(right - left, 4)
            height = round(bottom - top, 4)


            if image_idx == (total_img_num - 1) and j == (len(classes) - 1):
                new_p.write(
                    "{\"image_id\":" + str(orig_index) + ", \"category_id\":" + str(coco_id) + ", \"bbox\":[" + \
                    str(left) + ", " + str(top) + ", " + str(width) + ", " + str(height) + "], \"score\":" + str(
                        scores[j]) + "}")
            else:
                # print("corrected left, top, width, height", left, top, width, height)
                new_p.write(
                    "{\"image_id\":" + str(orig_index) + ", \"category_id\":" + str(coco_id) + ", \"bbox\":[" + \
                    str(left) + ", " + str(top) + ", " + str(width) + ", " + str(height) + "], \"score\":" + str(
                        scores[j]) + "},\n")
        image_idx += 1
    new_p.write("]")







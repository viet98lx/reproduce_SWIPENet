import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from models.keras_ssd512_skip import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator_test import Evaluator
import numpy as np
import warnings
import re
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import argparse
import random
args_parser = argparse.ArgumentParser()
args_parser.add_argument('--stage', type=str, default='NECMA',
                        help='stage training')
args_parser.add_argument('--model_names', type=str, default='ssd512_2013_adam16_0.0001_time3_epoch-01_loss-58.8138_val_loss-11.4791.h5', help='model name')
args_parser.add_argument('--batch_size', type=int, default=16)
args_parser.add_argument('--initial_epoch', type=int, default=0, help='initial epoch')
args_parser.add_argument('--final_epoch', type=int, default=120, help='final epoch')
args_parser.add_argument('--image_set_name', type=str, default='Main/test.txt', help='name of dataset to predict')
args = args_parser.parse_args()

try:
    import h5py
except ImportError:
    warnings.warn("'h5py' module is missing. The fast HDF5 dataset option will be unavailable.")
try:
    import json
except ImportError:
    warnings.warn("'json' module is missing. The JSON-parser will be unavailable.")
try:
    from bs4 import BeautifulSoup
except ImportError:
    warnings.warn("'BeautifulSoup' module is missing. The XML-parser will be unavailable.")
try:
    import pickle
except ImportError:
    warnings.warn(
        "'pickle' module is missing. You won't be able to save parsed file lists and annotations as pickled files.")

img_height = 512
img_width = 512
# set dataset_name as URPC2017 or URPC2019
dataset_name='URPC2019'
images_dir = './data/'+dataset_name+'/JPEGImages/'
annotations_dir = './data/'+dataset_name+'/Annotations/'
image_set_filename = './data/'+dataset_name+'/ImageSets/' + args.image_set_name

# sample_weights_dir = os.getcwd()+'/dataset/'+dataset_name
# if not os.path.exists(sample_weights_dir):
#     os.mkdir(sample_weights_dir)
if dataset_name=='URPC2019':
    n_classes = 4
    classes = ['seaurchin', 'starfish', 'seacucumber', 'scallop']
    pred_format = {'seacucumber': 2, 'seaurchin': 0, 'scallop': 3, 'starfish': 1}
else:
    n_classes = 3
    classes = ['seacucumber', 'seaurchin', 'scallop']
    pred_format = {'seacucumber': 0, 'seaurchin': 1, 'scallop': 2}

detection_mode = 'test'
model_mode = 'inference'
# modelnames=['ssd512_2013_adam16_0.0001_time3_epoch-01_loss-58.8138_val_loss-11.4791.h5', 'ssd512_2013_adam16.h5']
modelnames=re.split(',', args.model_names)
modelindex=1
for modelname in modelnames:
    K.clear_session() # Clear previous models from memory.
    model = ssd_512(image_size=(img_height, img_width, 3),
                    n_classes=n_classes,
                    mode=model_mode,
                    l2_regularization=0.0005,
                    scales=[0.04, 0.07, 0.15, 0.3, 0.45, 0.6],
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0]],
                    two_boxes_for_ar1=True,
                    steps=[4, 8, 16, 32, 64],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.01,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)

    weights_path= os.path.join(os.getcwd(), modelname)
    model.load_weights(weights_path, by_name=True)
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    dataset = DataGenerator()
    dataset.parse_yolo_text(images_dirs=[images_dir],
                      image_set_filenames=[image_set_filename],
                      annotations_dirs=[annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=False,
                      ret=False)

    evaluator = Evaluator(model=model,
                          modelindex=modelindex,
                          n_classes=n_classes,
                          data_generator=dataset,
                          model_mode=model_mode,
                          detection_mode=detection_mode)
    evaluator.set_name = re.split("/",image_set_filename)[-1]
    results = evaluator(img_height=img_height,
                        img_width=img_width,
                        batch_size=args.batch_size,
                        data_generator_mode='resize',
                        round_confidences=False,
                        matching_iou_threshold=0.5,
                        border_pixels='include',
                        sorting_algorithm='quicksort',
                        average_precision_mode='sample',
                        num_recall_points=11,
                        ignore_neutral_boxes=True,
                        return_precisions=True,
                        return_recalls=True,
                        return_average_precisions=True,
                        verbose=True)
    modelindex=modelindex+1
    nb_gt_per_class = evaluator.get_num_gt_per_class(
                             ignore_neutral_boxes=True,
                             verbose=True,
                             ret=True)
    print(nb_gt_per_class)

    # true_positives, false_positives, cumulative_true_positives, cumulative_false_positives
    tuple_results =  evaluator.match_predictions(
                      ignore_neutral_boxes=True,
                      matching_iou_threshold=0.5,
                      border_pixels='include',
                      sorting_algorithm='quicksort',
                      verbose=True,
                      ret=True)
    print("TP: {}".format(tuple_results[0]))
    print("FP: {}".format(tuple_results[1]))
    print("CTP: {}".format(tuple_results[2]))
    print("CFP: {}".format(tuple_results[3]))

    # cumulative_precisions, cumulative_recalls
    tuple_Prec_Recall = evaluator.compute_precision_recall(verbose=True, ret=True)
    print("CPrec: {}".format(tuple_Prec_Recall[0]))
    print("CRecall: {}".format(tuple_Prec_Recall[1]))

    average_precisions = evaluator.compute_average_precisions(mode='sample', num_recall_points=11, verbose=True, ret=True)
    print("Average Prec: {}".format(average_precisions))

    mean_average_precision = evaluator.compute_mean_average_precision(ret=True)
    print("MAP: {}".format(mean_average_precision))
print('Detection results of multiple models have been saved in SWEIPENetv2/dataset/')

# Ensembel the results of multiple models
def intersection_area_(boxes1, boxes2, coords='corners', mode='outer_product', border_pixels='half'):
    '''
    The same as 'intersection_area()' but for internal use, i.e. without all the safety checks.
    '''
    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`
    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1  # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1  # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.

    # Compute the intersection areas.
    if mode == 'outer_product':
        # For all possible box combinations, get the greater xmin and ymin values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmin, ymin]], axis=0), reps=(m, 1, 1)))
        # For all possible box combinations, get the smaller xmax and ymax values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmax, ymax]], axis=0), reps=(m, 1, 1)))

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:, :, 0] * side_lengths[:, :, 1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
        max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:, 0] * side_lengths[:, 1]

def iou(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2: raise ValueError(
        "boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError(
        "boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError(
        "All boxes must consist of 4 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(
            boxes1.shape[1], boxes2.shape[1]))

    # Compute the interesection areas.

    intersection_areas = intersection_area_(boxes1, boxes2, coords=coords, mode=mode)

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    # Compute the union areas.

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1  # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1  # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.

    if mode == 'outer_product':

        boxes1_areas = np.tile(
            np.expand_dims((boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d),
                           axis=1), reps=(1, n))
        boxes2_areas = np.tile(
            np.expand_dims((boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d),
                           axis=0), reps=(m, 1))

    elif mode == 'element-wise':

        boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d)
        boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas

def nms(boxes, overlap):
    if not boxes.shape[0]:
        pick = []
    else:
        trial = boxes
        x1 = trial[:, 0]
        y1 = trial[:, 1]
        x2 = trial[:, 2]
        y2 = trial[:, 3]
        score = trial[:, 4]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        I = np.argsort(score)
        pick = []
        count = 1
        while (I.size != 0):
            # print "Iteration:",count
            last = I.size
            i = I[last - 1]
            pick.append(i)
            suppress = [last - 1]
            for pos in range(last - 1):
                j = I[pos]
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                w = xx2 - xx1 + 1
                h = yy2 - yy1 + 1
                if (w > 0 and h > 0):
                    o = w * h / area[j]
                    if (o > overlap):
                        suppress.append(pos)
            I = np.delete(I, suppress)
            count = count + 1
    return pick

# datasetpath= os.path.join(os.getcwd(), 'dataset/Detections')
# filenames = []
# image_ids = []
# labels = []
# box_numbers = 0
# detections_set_dirs = []
#
# for (root, dirs, files) in os.walk(datasetpath):
#     for dir in dirs:
#         detections_set_dirs.append(dir)
#
# weight_set = [0.158, 0.332]
# with open(image_set_filename) as f:
#     image_id = [line.strip() for line in f]
#     image_ids += image_id
#
# results = [list() for _ in range(4)]
# for im in image_ids:
#     # print(im+'\n')
#     txtname = im + '.txt'
#     allboxes = []
#     allclses = []
#     allconfids = []
#     for detections_dir in detections_set_dirs:
#         with open(os.path.join(datasetpath, detections_dir, txtname)) as f:
#             boxes = []
#             clses = []
#             confids = []
#             for line in f:
#                 split_line = line.split()
#                 class_name = split_line[0]
#                 confid = float(split_line[1])
#                 xmin = float(split_line[2])
#                 ymin = float(split_line[3])
#                 xmax = float(split_line[4])
#                 ymax = float(split_line[5])
#                 box = [xmin, ymin, xmax, ymax]
#                 boxes.append(box)
#                 clses.append(class_name)
#                 confids.append(confid)
#         allclses.append(clses)
#         allboxes.append(boxes)
#         allconfids.append(confids)
#     allboxes = np.array(allboxes, dtype='float')
#     allconfids = np.array(allconfids, dtype='float')
#     for i in range(allboxes.shape[0]):
#         for j in range(allboxes.shape[1]):
#             curbox = allboxes[i, j]
#             # For curbox, construct a highly overlapped box set to vote for its cls and cooridiante.
#             set_box = []
#             set_cls = []
#             set_weight = []
#             set_confid = []
#             for k in range(allboxes.shape[0]):
#                 # Compute the IoU similarities between all anchor boxes and all ground truth boxes for this batch item.
#                 overlaps = iou(curbox, allboxes[k], coords='corners', mode='outer_product', border_pixels='half')
#                 # For each ground truth box, get the anchor box to match with it.
#                 # matches = match_multi(weight_matrix=similarities, threshold=0.5)
#                 gt_match_index = np.argmax(overlaps)
#                 set_box.append(allboxes[k][gt_match_index])
#                 set_cls.append(allclses[k][gt_match_index])
#                 set_weight.append(weight_set[k])
#                 set_confid.append(allconfids[k][gt_match_index])
#             # Use box set to vote for its score and cooridiante.
#             if dataset_name == 'URPC2019':
#                 cls_score = [0, 0, 0, 0]
#             else:
#                 cls_score = [0, 0, 0]
#             box_score = 0
#             weight_score = 0
#             for m in range(len(set_box)):
#                 # print(str(m)+set_cls[m])
#                 class_id = pred_format[set_cls[m]]
#                 cls_score[class_id] = cls_score[class_id] + set_weight[m] * set_confid[m]
#                 box_score = box_score + set_weight[m] * set_box[m]
#                 weight_score = weight_score + set_weight[m]
#             box_score = box_score / weight_score
#             allboxes[i, j] = box_score
#             allclses[i][j] = classes[np.argmax(cls_score)]
#             allconfids[i][j] = np.max(cls_score)
#
#     allboxes_concat = np.concatenate(allboxes, axis=0)
#     allclses_concat = np.concatenate(allclses, axis=0)
#     allconfids_concat = np.concatenate(allconfids, axis=0)
#     extend_allconfids = np.expand_dims(allconfids_concat, axis=1)
#     nms_preboxes = np.concatenate((allboxes_concat, extend_allconfids), axis=1)
#     nms_boxes_index = nms(nms_preboxes, 0.5)
#     nms_boxes = nms_preboxes[nms_boxes_index][:]
#     nms_clses = allclses_concat[nms_boxes_index]
#
#     seacucumber_index = np.where(nms_clses == 'seacucumber')
#     seacucumber_boxes = nms_boxes[seacucumber_index]
#     seacucumber_clsnames = nms_clses[seacucumber_index]
#     for l in range(np.shape(seacucumber_index)[1]):
#         seacucumber_imname = im
#         boxstr = im + ' ' + str(round(seacucumber_boxes[l][4], 4)) + ' ' + str(
#             int(seacucumber_boxes[l][0])) + ' ' + str(int(seacucumber_boxes[l][1])) + ' ' + str(
#             int(seacucumber_boxes[l][2])) + ' ' + str(int(seacucumber_boxes[l][3]))
#         results[2].append(boxstr)
#
#     seaurchin_index = np.where(nms_clses == 'seaurchin')
#     seaurchin_boxes = nms_boxes[seaurchin_index]
#     seaurchin_clsnames = nms_clses[seaurchin_index]
#     for l in range(np.shape(seaurchin_index)[1]):
#         boxstr = im + ' ' + str(round(seaurchin_boxes[l][4], 4)) + ' ' + str(
#             int(seaurchin_boxes[l][0])) + ' ' + str(int(seaurchin_boxes[l][1])) + ' ' + str(
#             int(seaurchin_boxes[l][2])) + ' ' + str(int(seaurchin_boxes[l][3]))
#         results[0].append(boxstr)
#
#     scallop_index = np.where(nms_clses == 'scallop')
#     scallop_boxes = nms_boxes[scallop_index]
#     scallop_clsnames = nms_clses[scallop_index]
#     for l in range(np.shape(scallop_index)[1]):
#         boxstr = im + ' ' + str(round(scallop_boxes[l][4], 4)) + ' ' + str(int(scallop_boxes[l][0])) + ' ' + str(
#             int(scallop_boxes[l][1])) + ' ' + str(int(scallop_boxes[l][2])) + ' ' + str(int(scallop_boxes[l][3]))
#         results[3].append(boxstr)
#
#     starfish_index = np.where(nms_clses == 'starfish')
#     starfish_boxes = nms_boxes[starfish_index]
#     starfish_clsnames = nms_clses[starfish_index]
#     for l in range(np.shape(starfish_index)[1]):
#         boxstr = im + ' ' + str(round(starfish_boxes[l][4], 4)) + ' ' + str(int(starfish_boxes[l][0])) + ' ' + str(
#             int(starfish_boxes[l][1])) + ' ' + str(int(starfish_boxes[l][2])) + ' ' + str(int(starfish_boxes[l][3]))
#         results[1].append(boxstr)
#
# # {'seacucumber': 2, 'seaurchin': 0, 'scallop': 3, 'starfish': 1}
# file_fid = open(datasetpath + '/seacucumber.txt', 'w')
# for onestr in results[2]:
#     boxstr = onestr
#     file_fid.write(boxstr + '\n')
# file_fid.close()
#
# file_fid = open(datasetpath + '/seaurchin.txt', 'w')
# for onestr in results[0]:
#     boxstr = onestr
#     file_fid.write(boxstr + '\n')
# file_fid.close()
#
# file_fid = open(datasetpath + '/scallop.txt', 'w')
# for onestr in results[3]:
#     boxstr = onestr
#     file_fid.write(boxstr + '\n')
# file_fid.close()
#
# file_fid = open(datasetpath + '/starfish.txt', 'w')
# for onestr in results[1]:
#     boxstr = onestr
#     file_fid.write(boxstr + '\n')
# file_fid.close()
#
# print('The detection results of the final ensemble model have been saved in dataset/Detections/')
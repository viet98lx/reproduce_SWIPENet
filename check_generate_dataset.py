import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from models.keras_ssd512_skip import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from eval_utils.average_precision_evaluator_test import Evaluator
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_data_generator_weight_v2 import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
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
args_parser.add_argument('--model_names', type=str,
                         default='ssd512_2013_adam16_0.0001_time3_epoch-01_loss-58.8138_val_loss-11.4791.h5',
                         help='model name')
args_parser.add_argument('--test_file_set', type=str, default='./data/URPC2019/ImageSets/Main/test.txt')
args_parser.add_argument('--batch_size', type=int, default=16, help='batch size')
args_parser.add_argument('--nb_sample', type=int, default=5)
# args_parser.add_argument('--initial_epoch', type=int, default=0, help='initial epoch')
# args_parser.add_argument('--final_epoch', type=int, default=120, help='final epoch')
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
dataset_name = 'URPC2019'
images_dir = './data/' + dataset_name + '/JPEGImages/'
annotations_dir = './data/' + dataset_name + '/Annotations/'
# image_set_filename = './data/'+dataset_name+'/ImageSets/Main/test.txt'
image_set_filename = args.test_file_set

# sample_weights_dir = os.getcwd()+'/dataset/'+dataset_name
# if not os.path.exists(sample_weights_dir):
#     os.mkdir(sample_weights_dir)
if dataset_name == 'URPC2019':
    n_classes = 4
    classes = ['seaurchin', 'starfish', 'seacucumber', 'scallop']
    pred_format = {'seacucumber': 2, 'seaurchin': 0, 'scallop': 3, 'starfish': 1}
else:
    n_classes = 3
    classes = ['seacucumber', 'seaurchin', 'scallop']
    pred_format = {'seacucumber': 0, 'seaurchin': 1, 'scallop': 2}

img_height = 512
img_width = 512
img_channels = 3
mean_color = [123, 117, 104]
swap_channels = [2, 1, 0]
scales_pascal = [0.04, 0.07, 0.15, 0.3, 0.45, 0.6]
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0]]
two_boxes_for_ar1 = True
steps = [4, 8, 16, 32, 64]
offsets = [0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True

dataset_name='URPC2019'
images_dir = './data/'+dataset_name+'/JPEGImages/'
annotations_dir = './data/'+dataset_name+'/Annotations/'
sample_weights_dir = os.getcwd()+'/dataset/'+dataset_name
if not os.path.exists(sample_weights_dir):
    os.mkdir(sample_weights_dir)
sample_weights_dir = sample_weights_dir+'/weights.txt'

K.clear_session()
model = ssd_512(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)
weights_path = args.model_names
model.load_weights(weights_path, by_name=True)
print(model.summary())
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

predictor_sizes = [model.get_layer('deconv3_2_mbox_conf').output_shape[1:3],
                   model.get_layer('deconv4_2_mbox_conf').output_shape[1:3],
                   model.get_layer('deconv5_2_mbox_conf').output_shape[1:3],
                   model.get_layer('deconv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_add_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)


dataset = DataGenerator()
dataset.parse_yolo_text(images_dirs=[images_dir],
                    image_set_filenames=[image_set_filename],
                    sample_weights_dirs=sample_weights_dir,
                    annotations_dirs=[annotations_dir],
                    classes=classes,
                    include_classes='all',
                    exclude_truncated=False,
                    exclude_difficult=False,
                    ret=False)

for i in range(args.nb_sample):
    test_gen_data = next(dataset.generate(
        batch_size=args.batch_size,
        shuffle=True,
        transformations=[ssd_data_augmentation],
        label_encoder=ssd_input_encoder,
        keep_images_without_gt=False,
        returns={'processed_images', 'encoded_labels', 'filenames', 'original_labels'}))
    print("Processed Image")
    # print(test_gen_data[0])
    print("Encoded labels")
    print(test_gen_data[1].shape)
    # print(test_gen_data[1][0])
    random_idxs = random.sample([i for i in range(test_gen_data[1].shape[1])], k=3)
    print(random_idxs)
    sum_01 =  (test_gen_data[1][0][random_idxs[0]] == test_gen_data[1][0][random_idxs[1]]).sum()
    print("NB elements Encoded labels are same 01 {}".format(sum_01))
    print(np.where(test_gen_data[1][0][random_idxs[0]] == test_gen_data[1][0][random_idxs[1]]))

    sum_02 = (test_gen_data[1][0][random_idxs[0]] == test_gen_data[1][0][random_idxs[2]]).sum()
    print("NB elements Encoded labels are same 02 {}".format(sum_02))
    print(np.where(test_gen_data[1][0][random_idxs[0]] == test_gen_data[1][0][random_idxs[2]]))

    sum_12 = (test_gen_data[1][0][random_idxs[1]] == test_gen_data[1][0][random_idxs[2]]).sum()
    print("NB elements Encoded labels are same 12 {}".format(sum_12))
    print(np.where(test_gen_data[1][0][random_idxs[1]] == test_gen_data[1][0][random_idxs[2]]))

    print("Filenames")
    print(test_gen_data[2])
    print("Original labels")
    print(test_gen_data[3])
    print("----------------")

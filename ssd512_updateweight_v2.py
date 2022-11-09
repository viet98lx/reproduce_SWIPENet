import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from models.keras_ssd512_skip import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from data_generator.object_detection_2d_data_generator_update import DataGenerator
from eval_utils.average_precision_evaluator_train import Evaluator
import re
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import argparse
import random
args_parser = argparse.ArgumentParser()
args_parser.add_argument('--stage', type=str, default='NECMA',
                        help='stage training')
args_parser.add_argument('--model_name', type=str, default='ssd512_2013_adam16_0.0001_time3_epoch-01_loss-58.8138_val_loss-11.4791.h5', help='model name')
args_parser.add_argument('--initial_epoch', type=int, default=0, help='initial epoch')
args_parser.add_argument('--final_epoch', type=int, default=120, help='final epoch')
args = args_parser.parse_args()


img_height = 512
img_width = 512
# set dataset_name as URPC2017 or URPC2019
dataset_name='URPC2019'
images_dir = './data/'+dataset_name+'/JPEGImages/'
annotations_dir = './data/'+dataset_name+'/Annotations/'
image_set_filename = './data/'+dataset_name+'/ImageSets/Main/trainval.txt'
sample_weights_dir = os.getcwd()+'/dataset/'+dataset_name
if not os.path.exists(sample_weights_dir):
    os.mkdir(sample_weights_dir)
if dataset_name=='URPC2019':
    n_classes = 4
    classes = ['background', 'seaurchin', 'starfish', 'seacucumber', 'scallop']
else:
    n_classes = 3
    classes = ['background', 'seacucumber', 'seaurchin', 'scallop']

# set which stage of the Curriculm Multiclass Adaboost (CMA) algorithm (NECMA or NLCMA)
stage=args.stage # NLCMA

model_mode = 'inference'
# set modelname as the model saved in the last iteration
modelname=args.model_name
K.clear_session()
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


weights_path = os.path.join(os.getcwd(), modelname)
model.load_weights(weights_path, by_name=True)
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

dataset = DataGenerator()
dataset.parse_yolo_text(images_dirs=[images_dir],
                  image_set_filenames=[image_set_filename],
                  sample_weights_dirs=[os.path.join(os.getcwd(),'dataset/'+dataset_name+'/')],
                  annotations_dirs=[annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

evaluator = Evaluator(model=model,
                      stage=stage,
                      sample_weights_dir=sample_weights_dir,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=32,
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
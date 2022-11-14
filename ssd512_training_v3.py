import tensorflow as tf

# print("Avai GPU:")
# from tensorflow.python.client import device_lib
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'Device found : {gpus}')
# print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.test.is_gpu_available()

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
import tensorflow.keras.backend as K
from math import ceil
from math import floor
from models.keras_ssd512_skip import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_data_generator_weight_v3 import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import argparse
import random
import re
args_parser = argparse.ArgumentParser()
args_parser.add_argument('--weights_path', type=str, default='VGG_ILSVRC_16_layers_fc_reduced.h5',
                        help='check point weights path')
args_parser.add_argument('--batch_size', type=int, default=16, help='batch size')
args_parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
args_parser.add_argument('--initial_epoch', type=int, default=0, help='initial epoch')
args_parser.add_argument('--final_epoch', type=int, default=120, help='final epoch')
args_parser.add_argument('--trainval_file_set', type=str,default='./data/URPC2019/ImageSets/Main/trainval.txt')
args_parser.add_argument('--test_file_set', type=str,default='./data/URPC2019/ImageSets/Main/test.txt')
args_parser.add_argument('--model_name', type=str,default='ssd512_adam_URPC2019')

args = args_parser.parse_args()

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
# set dataset_name as URPC2017 or URPC2019
dataset_name='URPC2019'
batch_size = args.batch_size

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)
# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None,
                              batch_size=batch_size,
                              mode='training',
                              transformations=[ssd_data_augmentation],
                              label_encoder=None,
                              )
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None,
                            batch_size=batch_size,
                            mode='validating',
                            transformations=[convert_to_3_channels,resize],
                            label_encoder=None,
                            )
images_dir = './data/'+dataset_name+'/JPEGImages/'
annotations_dir = './data/'+dataset_name+'/Annotations/'
sample_weights_dir = os.getcwd()+'/dataset/'+dataset_name
if not os.path.exists(sample_weights_dir):
    os.mkdir(sample_weights_dir)
sample_weights_dir = sample_weights_dir+'/weights.txt'
# trainval_image_set_filename = './data/'+dataset_name+'/ImageSets/Main/trainval.txt'
trainval_image_set_filename = args.trainval_file_set
# test_image_set_filename = './data/'+dataset_name+'/ImageSets/Main/test.txt'
test_image_set_filename = args.test_file_set

if dataset_name=='URPC2019':
    n_classes = 4
    classes = ['background', 'seaurchin', 'starfish', 'seacucumber', 'scallop']
    # randomly split the URPC2019 dataset into 1999 train set and 898 test set
    # imnames = os.listdir(annotations_dir)
    # if not os.path.exists(trainval_image_set_filename) and not os.path.exists(test_image_set_filename):
    #     trainfid = open(trainval_image_set_filename, 'w')
    #     testfid = open(test_image_set_filename, 'w')
    #     randnums=random.sample(range(1,len(imnames)),1999)
    #     for i in range(len(imnames)):
    #         if i in randnums:
    #             trainfid.writelines(imnames[i][0:-4] + '\n')
    #         else:
    #             testfid.writelines(imnames[i][0:-4] + '\n')
    #     testfid.close()
    #     trainfid.close()
else:
    n_classes = 3
    classes = ['background', 'seacucumber', 'seaurchin', 'scallop']
print(classes)
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

# weights_path = 'VGG_ILSVRC_16_layers_fc_reduced.h5'
weights_path = args.weights_path
model.load_weights(weights_path, by_name=True)
for layer in model.layers[:21]:
    layer.trainable = False
print(model.summary())
adam = Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
# If the weights.txt not exist, initial the weight of each sample as 1, and save them in  weights.txt
if not os.path.exists(sample_weights_dir):
    train_dataset.parse_weight(images_dirs=[images_dir],
                            image_set_filenames=[trainval_image_set_filename],
                            sample_weights_dirs=sample_weights_dir,
                            annotations_dirs=[annotations_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

train_dataset.parse_yolo_text(images_dirs=[images_dir],
                        image_set_filenames=[trainval_image_set_filename],
                        sample_weights_dirs=sample_weights_dir,
                        annotations_dirs=[annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)
val_dataset.parse_yolo_text(images_dirs=[images_dir],
                      image_set_filenames=[test_image_set_filename],
                      sample_weights_dirs=sample_weights_dir,
                      annotations_dirs=[annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)
# train_dataset.create_hdf5_dataset(file_path='dataset_'+dataset_name+'_trainval.h5',
#                                   resize=False,
#                                   variable_image_size=True,
#                                   verbose=True)
# val_dataset.create_hdf5_dataset(file_path='dataset_'+dataset_name+'_test.h5',
#                                 resize=False,
#                                 variable_image_size=True,
#                                 verbose=True)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
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

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.
# train_generator = train_dataset.generate(batch_size=batch_size,
#                                          shuffle=True,
#                                          transformations=[ssd_data_augmentation],
#                                          label_encoder=ssd_input_encoder,
#                                          returns={'processed_images',
#                                                   'encoded_labels',
#                                                   'sample_weights'},
#                                          keep_images_without_gt=False)
train_dataset.label_encoder = ssd_input_encoder
# val_generator = val_dataset.generate(batch_size=batch_size,
#                                      shuffle=False,
#                                      transformations=[convert_to_3_channels,
#                                                       resize],
#                                      label_encoder=ssd_input_encoder,
#                                      returns={'processed_images',
#                                               'encoded_labels'},
#                                      keep_images_without_gt=False)
val_dataset.label_encoder = ssd_input_encoder

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()
print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# Define a learning rate schedule.
def lr_schedule(epoch, lr):
    decay_rate = 0.1
    decay_step = 60
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

# Define model callbacks.
# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath=args.model_name+'.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True, # True
                                   save_weights_only=True,
                                   mode='auto',
                                   period=1)
csv_logger = CSVLogger(filename=args.model_name+'_training_log.csv',
                       separator=',',
                       append=True)
learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)
terminate_on_nan = TerminateOnNaN()
callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = args.initial_epoch
final_epoch = args.final_epoch
steps_per_epoch = floor(train_dataset_size/batch_size)
with tf.device('/gpu:0'):
    history = model.fit(train_dataset,
                      steps_per_epoch=steps_per_epoch,
                      epochs=final_epoch,
                      callbacks=callbacks,
                      validation_data=val_dataset,
                      validation_steps=floor(val_dataset_size/batch_size),
                      initial_epoch=initial_epoch)
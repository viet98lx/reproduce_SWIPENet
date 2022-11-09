

from __future__ import division
import numpy as np
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import cv2
import csv
import os
import sys
from tqdm import tqdm, trange
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
    warnings.warn("'pickle' module is missing. You won't be able to save parsed file lists and annotations as pickled files.")


def parse_weight(images_dirs,
              image_set_filenames,
              sample_weights_dirs = None,
              annotations_dirs=[],
              classes=['background', 'seacucumber', 'seaurchin', 'scallop', 'starfish']):
    # Set class members.
    self.images_dirs = images_dirs
    self.annotations_dirs = annotations_dirs
    self.sample_weights_dirs = sample_weights_dirs
    self.image_set_filenames = image_set_filenames
    self.classes = classes
    self.include_classes = include_classes

    # Erase data that might have been parsed before.
    self.filenames = []
    self.image_ids = []
    self.labels = []
    self.sample_weights = []
    self.eval_neutral = []
    if not annotations_dirs:
        self.labels = None
        self.eval_neutral = None
        annotations_dirs = [None] * len(images_dirs)

    for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
        # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
        with open(image_set_filename) as f:
            image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
            self.image_ids += image_ids

        if verbose: it = tqdm(image_ids, desc="Processing image set '{}'".format(os.path.basename(image_set_filename)), file=sys.stdout)
        else: it = image_ids
        with open('/data/deeplearn/VOCdevkit/URPC2019/weights.txt') as f:
            weights = [line.strip().split(',') for line in f]
            gtboxlist=[box[0] for box in weights]
            gtweightlist=[box[1] for box in weights]

        # Loop over all images in this dataset.
        for image_id in it:
            filename = '{}'.format(image_id) + '.jpg'
            self.filenames.append(os.path.join(images_dir, filename))

            if not annotations_dirs is None:
                # Parse the XML file for this image.
                with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                    soup = BeautifulSoup(f, 'xml')

                folder = soup.folder.text # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
                #filename = soup.filename.text

                boxes = [] # We'll store all boxes for this image here.
                eval_neutr = [] # We'll store whether a box is annotated as "difficult" here.
                objects = soup.find_all('object') # Get a list of all objects in this image.

                # Parse the data for each object.
                for obj in objects:
                    class_name = obj.find('name', recursive=False).text
                    class_id = self.classes.index(class_name)
                    # Check whether this class is supposed to be included in the dataset.
                    if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
                    pose = obj.find('pose', recursive=False).text
                    truncated = int(obj.find('truncated', recursive=False).text)
                    if exclude_truncated and (truncated == 1): continue
                    difficult = int(obj.find('difficult', recursive=False).text)
                    if exclude_difficult and (difficult == 1): continue
                    # Get the bounding box coordinates.
                    bndbox = obj.find('bndbox', recursive=False)
                    xmin = int(bndbox.xmin.text)
                    ymin = int(bndbox.ymin.text)
                    xmax = int(bndbox.xmax.text)
                    ymax = int(bndbox.ymax.text)
                    item_dict = {'folder': folder,
                                 'image_name': filename,
                                 'image_id': image_id,
                                 'class_name': class_name,
                                 'class_id': class_id,
                                 'pose': pose,
                                 'truncated': truncated,
                                 'difficult': difficult,
                                 'xmin': xmin,
                                 'ymin': ymin,
                                 'xmax': xmax,
                                 'ymax': ymax}

                    if not sample_weights_dirs is None:
                        boxstr = image_id + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)
                        index = gtboxlist.index(boxstr)
                        weight = float(gtweightlist[index])
                    box = []
                    if not sample_weights_dirs is None:
                        box.append(weight)
                    else:
                        box.append(1)
                    for item in self.labels_output_format:
                        box.append(item_dict[item])
                    boxes.append(box)

                    if difficult: eval_neutr.append(True)
                    else: eval_neutr.append(False)

                self.labels.append(boxes)
                self.eval_neutral.append(eval_neutr)

    self.dataset_size = len(self.filenames)
    self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
    if self.load_images_into_memory:
        self.images = []
        if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
        else: it = self.filenames
        for filename in it:
            with Image.open(filename) as image:
                self.images.append(np.array(image, dtype=np.uint8))

    if ret:
        return self.images, self.filenames, self.labels, self.image_ids, self.eval_neutral

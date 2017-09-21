from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data_set import DataSet


import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import pickle
import numpy
import cv2
numpy.set_printoptions(threshold=numpy.nan)
import csv
from PIL import Image
from sklearn.preprocessing import normalize
from scipy import misc
from imgaug import augmenters as iaa
import random

#from resizeimage import resizeimage

#from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


pt_test_list_x = []
pt_test_list_y = []
test_x2 = []
test_y2 = []
    # sess.run(tf.global_variables_initializer())
for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/T1_GD_testing"):
  # if num_of_pts >= 400: #do 200 more slices
  #   break
  for f in filenames:
    print(os.path.join(root, f))
    # if num_of_pts >= 400:
    #   break

    data = pickle.load(open(os.path.join(root, f), "rb"))

    patient_full_name = f.split(".")[0]
    patient_type = 0
    patient_name = patient_full_name.split("_")[1]
    patient_found = False
    # if 'LGG' in f:
    #     print("&&&&&&&&")
    #     patient_type = 2
    if patient_type == 1 or patient_type == 0:
      csv_file = csv.reader(open('/data/ak4966/MRI/data/TCGA_id_list.csv', "rt"), delimiter=",")
      for row in csv_file:
        if row[0] == patient_name:

          patient_found = True
          print(row[4])
          if row[4] == "WT":
            patient_type = 0
          elif row[4] == "Mutant":
            patient_type = 1;
          elif row[4] == "NA":
            patient_type = 2;
      if patient_found == False:
        continue
    print(data.shape)     
    test_x1 = []  
    test_y1 = []
    for p in range(data.shape[2]):

      # im = Image.fromarray(data[:,:,p])
      # im.convert("L").save("test.jpeg")

      # image = Image.open("test.jpeg")
    # image.show()
      yy = misc.imresize(data[:,:,p], size=(256,256), interp = 'nearest')
  # image = image.resize((256, 256), Image.ANTIALIAS)
  # resized_image = resizeimage.resize_contain(image, [256, 256])
  # resized_image.convert('RGB').save("new_image.jpeg", image.format)

  # new_image = Image.open("new_image.jpeg")
  # new_image.show()
      arr = numpy.asarray(yy)

      
      temp2 = numpy.reshape(arr, 65536)
      if len(test_x1) == 0:
        # normed = normalize(arr, axis=0, norm='l2')
        # test_x1 = numpy.reshape(normed, 65536)
        test_x1 = temp2
      else:
        # normed = normalize(arr, axis=0, norm='l2')
        # test_x1 = numpy.vstack([test_x1, numpy.reshape(normed, 65536)])
        test_x1 = numpy.vstack([test_x1, temp2])
      
      if patient_type == 0:
        if len(test_y1) == 0:
          test_y1 = [1, 0, 0]
        else:
          test_y1 = numpy.vstack([test_y1, [1, 0, 0]])
      elif patient_type == 1:
        if len(test_y1) == 0:
          test_y1 = [0, 1, 0]
        else:
          test_y1 = numpy.vstack([test_y1, [0, 1, 0]])
      elif patient_type == 2:
        if len(test_y1) == 0:
          test_y1 = [0, 0, 1]
        else:
          test_y1 = numpy.vstack([test_y1, [0, 0, 1]])
    # pt_test_list_x.append(test_x1)
    test_x2.append((f,) + tuple(numpy.asarray(test_x1)))
    # pt_test_list_y.append(test_y1)
    test_y2.append((f,) + tuple(numpy.asarray(test_y1)))
    print(patient_type)



# test_x2 = numpy.asarray(pt_test_list_x)
# test_y2 = numpy.asarray(pt_test_list_y)

pickle.dump(test_x2, open( "T1_GD_testing_x_no_normalization_no_aggregation.p", "wb" ), protocol=4)
pickle.dump(test_y2, open( "T1_GD_testing_y_no_normalization_no_aggregation.p", "wb" ), protocol=4)
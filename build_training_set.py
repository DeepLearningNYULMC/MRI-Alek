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


  # Import data
  ###mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

shuffle_x = []
x1 = []
y1 = []
num_of_pts = 0

batch_list = []

for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/T1_GD_training/train"):
# if num_of_pts >= 200: #only do 200 slices
#   break
	for f in filenames:
	  print(os.path.join(root, f))
	  print("num pts " + str(num_of_pts))
	  # if num_of_pts >= 200:
	  #   break
	  data = pickle.load(open(os.path.join(root, f), "rb"))

	  patient_full_name = f.split(".")[0]
	  patient_type = 0
	  patient_name = patient_full_name.split("_")[1]
	  patient_found = False
	  # if 'LGG' in f:
	  # 	print("&&&&&&&&")
	  # 	patient_type = 2
	  if patient_type == 1 or patient_type == 0:
	    csv_file = csv.reader(open('/data/ak4966/MRI/data/TCGA_id_list.csv', "rt"), delimiter=",")
	    for row in csv_file:
	      if row[0] == patient_name:
	        patient_found = True
	        print(row[4])

	        if row[4] == "WT":
	          
	          print("@@@@@@@@")
	          patient_type = 0
	        elif row[4] == "Mutant":
	          
	          print("########")
	          patient_type = 1
	        elif row[4] == "NA":
	          print("&&&&&&&&")
	          patient_type = 2
	     	
	    if patient_found == False:
	      continue
	  print(data.shape)        
	  for p in range(data.shape[2]):

	    # print(data[:,:,p].shape)

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
	    # print(arr.shape)


	    temp = numpy.reshape(arr, 65536)
	    x1.append(temp)
	    num_of_pts = num_of_pts + 1
	    print("num after " + str(num_of_pts))
	    if patient_type == 0:
	      y1.append([1, 0, 0])
	    elif patient_type == 1:
	      y1.append([0, 1, 0])
	    elif patient_type == 2:
	      y1.append([0, 0, 1])
	  print(patient_type)


#shuffling code
# x2 = []
# print(x1[0])
# for i in range(len(x1)):

# 	normed = normalize(x1[i], axis=0, norm='l2')
# 	print(i)
# 	plt.imshow(normed)
# 	x2.append(numpy.reshape(normed, 65536))


# print("189")
# print(x2[0])
# print("here")

x2 = numpy.asarray(x1)
y2 = numpy.asarray(y1)
print(y2)

validate_x1 = []
validate_y1 = []
for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/new_data/T1_GD_training/validation"):
# if num_of_pts >= 200: #only do 200 slices
#   break
	for f in filenames:
	  print(os.path.join(root, f))
	  print("num pts " + str(num_of_pts))
	  # if num_of_pts >= 200:
	  #   break
	  data = pickle.load(open(os.path.join(root, f), "rb"))

	  patient_full_name = f.split(".")[0]
	  patient_type = 0
	  patient_name = patient_full_name.split("_")[1]
	  patient_found = False
	  # if 'LGG' in f:
	  # 	print("&&&&&&&&")
	  # 	patient_type = 2
	  if patient_type == 1 or patient_type == 0:
	    csv_file = csv.reader(open('/data/ak4966/MRI/data/TCGA_id_list.csv', "rt"), delimiter=",")
	    for row in csv_file:
	      if row[0] == patient_name:
	        patient_found = True
	        print(row[4])

	        if row[4] == "WT":
	          
	          print("@@@@@@@@")
	          patient_type = 0
	        elif row[4] == "Mutant":
	          
	          print("########")
	          patient_type = 1
	        elif row[4] == "NA":
	          print("&&&&&&&&")
	          patient_type = 2

	    if patient_found == False:
	      continue
	  print(data.shape)        
	  for p in range(data.shape[2]):

	    # print(data[:,:,p].shape)

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

	    # print(arr.shape)


	    temp = numpy.reshape(arr, 65536)
	    validate_x1.append(temp)
	    num_of_pts = num_of_pts + 1
	    print("num after " + str(num_of_pts))
	    if patient_type == 0:
	      validate_y1.append([1, 0, 0])
	    elif patient_type == 1:
	      validate_y1.append([0, 1, 0])
	    elif patient_type == 2:
	      validate_y1.append([0, 0, 1])
	  print(patient_type)


# validate_x2 = []
# for i in range(len(validate_x1)):

# 	normed = normalize(validate_x1[i], axis=0, norm='l2')
# 	print(i)
# 	plt.imshow(normed)
# 	validate_x2.append(numpy.reshape(normed, 65536))

#shuffling code
validate_x2 = numpy.asarray(validate_x1)
validate_y2 = numpy.asarray(validate_y1)
print(validate_y2)

# for q in range(len(x2)):
#   numpy.reshape(x2[q], 65536)

print(x2.shape)
print(y2.shape)

pickle.dump(x2, open( "T1_GD_all_x_no_normalization.p", "wb" ), protocol=4)
pickle.dump(y2, open( "T1_GD_all_y_no_normalization.p", "wb" ), protocol=4)
pickle.dump(validate_x2, open( "T1_GD_all__validation_x_no_normalization.p", "wb" ), protocol=4)
pickle.dump(validate_y2, open( "T1_GD_all__validation_y_no_normalization.p", "wb" ), protocol=4)


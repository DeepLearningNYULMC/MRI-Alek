from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
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
import numpy as np
def main(_):
    dataset_type_list = ['testing']
    modality_type_list = ['T1','T1_GD', 'T2_FLAIR','T1_FLAIR','T2']
    #target_dir = '/scratch/nsr3'
    target_dir = '/data/ak4966/MRI/data/pickle_files/'
    root_dir = '/data/ak4966/MRI/data/new_data/'
    csv_file_loc = '/data/ak4966/MRI/data/TCGA_id_list.csv'
    for modality_type in modality_type_list:
      print('type:', modality_type)
      for dataset_type in dataset_type_list:
        print('dataset group:', dataset_type)
        x1 = np.zeros((1,256*256), dtype=float)
        y1 = np.zeros((1,3), dtype=int)
        ids_1 = ['dummy']
        num_of_pts = 0
        for root, dirs, filenames in os.walk(root_dir+'/'+modality_type+"_testing/"):
            for f in filenames:
                print(os.path.join(root, f))
                print("num pts " + str(num_of_pts))
                data = pickle.load(open(os.path.join(root, f), "rb"))
                patient_full_name = f.split(".")[0]
                patient_type = 0
                patient_name = patient_full_name.split("_")[1]
                patient_found = False
                if patient_type == 1 or patient_type == 0:
                    csv_file = csv.reader(open(csv_file_loc, "rt"), delimiter=",")
                for row in csv_file:
                    if row[0] == patient_name:
                        patient_found = True
                        print('patient,', patient_name, 'is of type:',row[4])
                        if row[4] == "WT":
                           patient_type = 0
                        elif row[4] == "Mutant":
                           patient_type = 1
                        elif row[4] == "NA":
                           patient_type = 2
                if patient_found == False:
                   print('patient ', patient_name, 'in the CSV file. ')
                   continue
                print('patient data of shape:', data.shape)

                data_resized_slices_x = np.zeros((data.shape[2], 256*256), dtype=float)
                data_resized_slices_y = np.zeros((data.shape[2], 3), dtype=int)
                for p in range(data.shape[2]):
                    yy = misc.imresize(data[:,:,p], size=(256,256), interp = 'nearest')
                    data_resized_slices_x[p,:] = numpy.asarray(yy).reshape(1,256*256)
                    data_resized_slices_y[p, patient_type] = 1
                    ids_1.append(patient_name)
                    num_of_pts = num_of_pts + 1
                x1 = np.vstack([x1, data_resized_slices_x])
                y1 = np.vstack([y1, data_resized_slices_y])
                print('added', p, 'new slices and data is now of size:', x1.shape, y1.shape)
        print('----------------------------------------------')
        print('done! saving files in ',target_dir+'/'+ dataset_type + "_" +modality_type)
        pickle.dump(x1, open(target_dir+'/'+ dataset_type + "_" +modality_type+"_all_x_no_normalization.pkl", "wb" ), protocol=-1)
        pickle.dump(y1, open(target_dir+'/'+ dataset_type + "_" +modality_type+"_all_y_no_normalization.pkl", "wb" ), protocol=-1)
        pickle.dump(ids_1, open(target_dir+'/'+ dataset_type + "_" +modality_type+"_all_IDs_no_normalization.pkl", "wb" ), protocol=-1)
        print('success saving')


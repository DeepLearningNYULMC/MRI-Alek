

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
from imgaug import augmenters as iaa
import random

#from resizeimage import resizeimage
from sklearn.metrics import roc_curve, auc

#from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def random_alteration(x):

  seq = iaa.OneOf([
    iaa.Affine(rotate=random.randint(-2, 2)),
    iaa.Affine(scale={"x": (1.05, 0.95), "y": (1.05, 0.95)})
  ])
  return seq.augment_image(x)


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 256, 256, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([64 * 64 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 64*64*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 3])
  b_fc2 = bias_variable([3])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2], max_to_keep=0)
  return y_conv, keep_prob, saver

#Weight Initialization functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Convolution and Pooling functions (we can we go over what the 'strides' part is)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main(_):
  # Import data
  ###mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  print("starting to load data...")
  x2 = pickle.load(open( "T1_GD_all_x_no_normalization.p", "rb" ))
  print("x2 loaded.")
  y2 = pickle.load(open( "T1_GD_all_y_no_normalization.p", "rb" ))
  print("y2 loaded.")
  validate_x2 = pickle.load(open( "T1_GD_validation_x_no_normalization_aggregated_.p", "rb" ))
  print("validate_x2 loaded.")
  validate_y2 = pickle.load(open( "T1_GD_validation_y_no_normalization_aggregated.p", "rb" ))
  print("validate_y2 loaded.")
  validate_x2_nonaggregated = pickle.load(open( "T1_GD_all__validation_x_no_normalization.p", "rb" ))
  print("validate_x2 loaded.")
  validate_y2_nonaggregated = pickle.load(open( "T1_GD_all__validation_y_no_normalization.p", "rb" ))
  print("validate_y2 loaded.")

  number_epochs = sys.argv[0]
  kernal_size = sys.argv[2]


  data_set_all = DataSet(x2,y2, fake_data=False)
  validation_set_all = DataSet(validate_x2_nonaggregated, validate_y2_nonaggregated, fake_data=False)


  # Create the convolutional model
  x = tf.placeholder(tf.float32, [None, 65536])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 3])

  # Build the graph for the deep net
  # with tf.device('/gpu:2'):
  y_conv, keep_prob, saver = deepnn(x)
  print(keep_prob)


  #plt.imshow(mnist.test.images[0].reshape(28,28))
  #print(type(mnist.test.images))
  #print(mnist.test.images.shape)
  #plt.show()
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




  #grads = new_optimizer.compute_gradients(cross_entropy)
  data_points = []
  avg_loss = 0
  total_loss = 0
  avg_validation_loss = 0
  total_validation_loss = 0
  batch_size = sys.argv[1]
  batches_completed = 0
  validation_batches_completed = 0
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  output_file= open("T1_GD_validation_loss_file_no_normalization_"+epoch_size+"_epochs_"+kernal_size+"_kernalsize_"+batch_size+"_batchsize.txt","w+")
  
  with tf.Session(config = config) as sess:
    
    sess.run(tf.global_variables_initializer())


    # sess.graph.finalize()

    for i in range(5000):
      batch_x, batch_y = data_set_all.next_batch(batch_size)
      for batch_slice in batch_x:
        batch_slice = numpy.reshape(batch_slice, (256, 256))
        batch_slice = random_alteration(batch_slice)
        batch_slice = numpy.reshape(batch_slice, 65536)


      batches_completed += 1
      loss = sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
      total_loss += loss
      new_avg_loss = total_loss/batches_completed

      if(new_avg_loss>avg_loss and batches_completed != 1):
        avg_loss = new_avg_loss
      # break

      avg_loss = new_avg_loss

      data_points.append(loss)

      if i % 1000 == 0:
        validation_batch_x, validation_batch_y = validation_set_all.next_batch(batch_size)
        validation_batches_completed+=1
        train_accuracy = accuracy.eval(feed_dict={x: validation_batch_x, y_: validation_batch_y, keep_prob: 1.0})
        validation_loss = cross_entropy.eval(feed_dict={x: validation_batch_x, y_: validation_batch_y, keep_prob: 1.0})
        total_validation_loss += validation_loss
        new_avg_validation_loss = total_validation_loss/validation_batches_completed

        if(new_avg_validation_loss>avg_validation_loss and batches_completed!=1):
          avg_validation_loss = new_avg_validation_loss


        avg_validation_loss = new_avg_validation_loss

        output_file.write("Validation loss at i = %d is %g\n" % (i, avg_validation_loss))


        total_times = 0.0
        total_accuracy = 0.0
        prediction=tf.argmax(y_conv,1)
        probabilities=tf.nn.softmax(y_conv)
        probs_array = []
        condensed_y = []

        for j in range(len(validate_x2)):
          #print(test_x2[i])
          #print(test_y2[i])
          temp3 = accuracy.eval(feed_dict={x: validate_x2[j], y_: validate_y2[j], keep_prob: 1.0})
          print('test accuracy %g' % temp3)
          total_accuracy = total_accuracy + temp3
          total_times = total_times+1
          temp4 = prediction.eval(feed_dict={x: validate_x2[j], keep_prob: 1.0}, session=sess)
          print("predictions", temp4)
          probability = probabilities.eval(feed_dict={x: validate_x2[j], keep_prob: 1.0}, session=sess)
          print(probability)
          if j==0:
            probs_array = probability.mean(axis=0)
            condensed_y = validate_y2[j].mean(axis=0)
            continue
          probs_array = numpy.vstack([probs_array, probability.mean(axis=0)])
          condensed_y = numpy.vstack([condensed_y, validate_y2[j].mean(axis=0)])


        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for j in range(3):
          fpr[j], tpr[j], _ = roc_curve(condensed_y[:, j], probs_array[:, j])
          roc_auc[j] = auc(fpr[j], tpr[j])

          # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(condensed_y.ravel(), probs_array.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        output_file.write("ROCs at i = %d are "%i)
        for j in range(3):
          plt.plot(fpr[j], tpr[j], label='ROC curve of class {0} (area = {1:0.2f})'''.format(j, roc_auc[j]))
          output_file.write(str(roc_auc[j])+", ")

        output_file.write("\n")
        output_file.flush()
        print('step %d, training accuracy %g' % (i, train_accuracy))
        name = 'T1_GD_testing_with_intermediateROC_no_normalization_epoch_' + str(i)
        save_path = saver.save(sess, name)

      train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})


    #testing  
    print(avg_loss)
    output_file.close()
    save_path = saver.save(sess, 'T1_GD_testing_with_intermediateROC_no_normalization_final')
  #making a new session for testing
  # with tf.Session(config = config) as sess2:
    # test_x1 =[]
    # test_y1 =[]
    # # sess.run(tf.global_variables_initializer())
    # for root, dirs, filenames in os.walk("/data/ak4966/MRI/data/testing"):
    #   # if num_of_pts >= 400: #do 200 more slices
    #   #   break
    #   for f in filenames:
    #     print(os.path.join(root, f))
    #     # if num_of_pts >= 400:
    #     #   break

    #     data = pickle.load(open(os.path.join(root, f), "rb"))

    #     patient_full_name = f.split(".")[0]
    #     patient_type = int(patient_full_name.split("_")[1])
    #     patient_name = patient_full_name.split("_")[0]
    #     patient_found = False
    #     if patient_type == 1 or patient_type == 0:
    #       csv_file = csv.reader(open('/data/ak4966/MRI/data/TCGA_id_list.csv', "rt"), delimiter=",")
    #       for row in csv_file:
    #         if row[0] == patient_name:

    #           patient_found = True
    #           print(row[4])
    #           if row[4] == "WT":
    #             patient_type = 0
    #           elif row[4] == "Mutant":
    #             patient_type = 1;
    #           elif row[4] == "NA":
    #             patient_type = 2;
    #       if patient_found == False:
    #         continue
    #     print(data.shape)        
    #     for p in range(data.shape[2]):

    #       im = Image.fromarray(data[:,:,p])
    #       im.convert("L").save("test.jpeg")

    #       image = Image.open("test.jpeg")
    #     # image.show()
    #       image = image.resize((256, 256), Image.ANTIALIAS)
    #     # resized_image = resizeimage.resize_contain(image, [256, 256])
    #     # resized_image.convert('RGB').save("new_image.jpeg", image.format)

    #     # new_image = Image.open("new_image.jpeg")
    #     # new_image.show()
    #       arr = numpy.asarray(image)

    #       temp = numpy.reshape(arr, 65536)
    #       test_x1.append(temp)
    #       num_of_pts = num_of_pts +1
    #       if patient_type == 0:
    #         test_y1.append([1, 0, 0])
    #       elif patient_type == 1:
    #         test_y1.append([0, 1, 0])
    #       elif patient_type == 2:
    #         test_y1.append([0, 0, 1])
    #     print(patient_type)

    # test_x2 = numpy.asarray(test_x1)
    # test_y2 = numpy.asarray(test_y1)


    # print(test_x2.shape)
    # print(test_y2.shape)
    # # data_set_test_all = DataSet(test_x2,test_y2, fake_data=False)

    # print('test accuracy %g' % accuracy.eval(feed_dict={x: test_x2[0:1000], y_: test_y2[0:1000], keep_prob: 1.0}))

    # for q in range (len(test_x2)-1):
    #   test_x_batch, test_y_batch = data_set_test_all.next_batch(1)
    #   print('test accuracy %g' % accuracy.eval(feed_dict={x: test_x_batch, y_: test_y_batch, keep_prob: 1.0}))
#train
  
  # for i in range(10): 
  #   batch_x, batch_y = data_set_all.next_batch(batch_size)
  #   batches_completed += 1

  #   print(batch_y)

  #   # print("working...")

  #   loss = sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y})
  #   # print(loss)
  #   total_loss += loss
  #   new_avg_loss = total_loss/batches_completed
  #   print(new_avg_loss)
  #   print(avg_loss)

  #   if(new_avg_loss>avg_loss and batches_completed != 1):
  #     avg_loss = new_avg_loss
  #     # break

  #   avg_loss = new_avg_loss

  #   data_points.append(loss)
    

  #   sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

#get testing data
  # plt.plot(data_points)
  # plt.show()

      
  # print(sess.run(accuracy, feed_dict={x: test_x2, y_: test_y2}))
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


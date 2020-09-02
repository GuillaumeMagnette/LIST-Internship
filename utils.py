import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import l2_normalize
from tensorflow.keras import Input, Model

from tensorflow.keras.models import load_model, save_model

from numpy import linalg as LA

import urllib

#import tensorflow_docs as tfdocs
#import tensorflow_docs.plots

import tensorflow_datasets as tfds

import PIL.Image

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 5)

import skimage
from skimage import measure
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage import transform
from skimage.transform import rotate, AffineTransform, swirl, resize
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage import io, img_as_float, img_as_float64
from scipy import ndimage
from pathlib import Path
from skimage import data, img_as_float
from skimage import exposure

from random import random, uniform

import os
import pathlib
import shutil
import pandas as pd
#import cv2


import numba
from numba import jit

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt

from os import path                      # os level path manipulation
from glob import glob                    # Unix style pathname pattern expansion
import numpy as np                       # array goodnes
from matplotlib import pyplot as plt     # plotting library
#import nibabel as nib                    # handlie NIFTI files
from tqdm import tqdm, trange            # progress bars
from tensorflow.keras.utils import get_file  # handy function to download data

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import skimage.draw
import pandas as pd
import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

from skimage.transform import resize             # resize images
from skimage.exposure import equalize_adapthist  # CLAHE
from skimage.exposure import rescale_intensity   # used to normalize the image data
from sklearn.model_selection import train_test_split    # helper function to split the data
from skimage.morphology import area_closing, area_opening, binary_closing, binary_opening, black_tophat, closing, opening, skeletonize
import skimage
from skimage import measure
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage import transform
from skimage.transform import rotate, AffineTransform, swirl, resize
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage import io, img_as_float, img_as_float64
from scipy import ndimage
from pathlib import Path
from skimage import data, img_as_float
from skimage import exposure
from scipy.ndimage import shift


def random_saturation(rgb, show_result = False):
  """ Modify randomly the saturation of the input image """
  hsv = skimage.color.rgb2hsv(rgb)

  #plt.imshow(hsv)
  saturation = uniform(-0.35, 0.35)
  
  #print(hsv[:,:,1])
  hsv[:,:,1] = hsv[:,:,1] + saturation
  #print(np.max(lab[:,:,0]))
  #print(hsv[:,:,1])
  rgb = skimage.color.hsv2rgb(hsv)

  if show_result:
    plt.imshow(rgb)
    plt.show()

  return rgb

def random_brightness(rgb, show_result = False):
  """ Modify randomly the brightness of the input image """
  lab = skimage.color.rgb2lab(rgb)

  #plt.imshow(lab)
  brightness = uniform(-20, 20)
  
  #print(lab[:,:,0])
  lab[:,:,0] = lab[:,:,0] + brightness
  #print(np.max(lab[:,:,0]))
  #print(lab[:,:,0])
  rgb = skimage.color.lab2rgb(lab)
  if show_result:
    plt.imshow(rgb)
    plt.show()

  return rgb

def preprocess_images(image_path):
  """ apply all augmentation methods defined above on the image """

  image = img_as_float(io.imread(image_path))
  image = exposure.equalize_adapthist(image, clip_limit=0.03)
  image = random_saturation(image, show_result=False)
  image = random_brightness(image, show_result=False)
     
      
  image = image*255
      
  shear = uniform(-.1, .1)
  tfr = AffineTransform(shear=shear)
  sheared = transform.warp(image, tfr, order=1, preserve_range=True,mode='constant', cval=255)
  
  strength = uniform(-1, 1)
  image = swirl(sheared, rotation=0, strength=strength, radius=500,mode='constant', cval=255)


  return image

def augmentationImageUnet(img_path,img_aug_path, input_shape):

  """ Create a proper dataset where each folder contains the images of one 
  specimen, randomly augmented 20 times for each "real" image present in the 
  original dataset """

  #!rm -rf /content/databaseAug #useful when running the method again for makedirs
  data = pd.read_csv('/content/Stage_Tritons/training.csv')

  labels = data.iloc[:,4]

  names = data.iloc[:,0]

  labels_uniques, counts = np.unique(labels, return_counts=True)
  

  for label in labels_uniques:
    os.makedirs(img_aug_path+'/'+label)


  print("Création de la base de données augmentées en cours...")
  # List all files in a directory using scandir()
  basepath = img_path
  with os.scandir(basepath) as images:
    for im in images:

      if(im.name in names.values):
        
        id = names[names == im.name].index[0]
  
        label = labels.iloc[id]

        id_label = np.where(labels_uniques == label)

        imagePath = basepath + '/' + im.name

        count = counts[id_label]
        
        k = 0
        while (k < 20):  
          image = preprocess_images(imagePath)
          
          augPath = img_aug_path+'/'+label+'/' + im.name[:-4] + str(k) + '.jpg'
          image = tf.keras.preprocessing.image.img_to_array(image)
          plt.imsave(augPath, image/255)
          k += 1

def get_label_csv(file_path):
  parts = tf.strings.split(file_path, os.path.sep)

  name = parts[-1]

  id = names[names == name].index[0]
  
  label = labels.iloc[id]

  return label == CLASS_NAMES


def process_path_csv(file_path):
  #print(file_path)
  label = get_label_csv(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  try:
    img = decode_img(img)
    #print(np.max(img))
    print("decoded")
    #counter +=1
  except:
    print("erreur décodage")

  return img, label


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def process_path_new(file_path):
  #print(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  try:
    img = decode_img(img)
    #print(np.max(img))
    print("decoded")
    counter +=1
  except:
    print("erreur décodage")

  return img

def format_image_new(image):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (input_shape[0], input_shape[1]))
  return image


def augmentationImages(img_path,img_aug_path):

  """ Create a proper dataset where each folder contains the images of one 
  specimen, randomly augmented 20 times for each "real" image present in the 
  original dataset """

  #!rm -rf /content/databaseAug #useful when running the method again for makedirs
  #if (len(os.listdir(img_aug_path)) == 0):
  #with open('dataset.csv', 'rb') as f:
   # data = f.read()

  data = pd.read_csv('dataset.csv')

  labels = data.iloc[:,1]

  names = data.iloc[:,0]

  labels_uniques, counts = np.unique(labels, return_counts=True)

  #list_images = os.listdir(img_path)
  #labels_uniques = len(list_images) #We assume newts from the first session are all different
  

  for label in labels_uniques:
    os.makedirs(img_aug_path + '/' + str(label))
 

  print("Creating the augmented dataset...")
  # List all files in a directory using scandir()
  basepath = img_path
  with os.scandir(basepath) as images:
    for im in images:

      if(im.name in names.values):
        
        id = names[names == im.name].index[0]
  
        label = labels.iloc[id]

        id_label = np.where(labels_uniques == label)

        imagePath = basepath + '/' + im.name

        count = counts[id_label]
        
        k = 0
        while (k < 20 // count):  
          image = preprocess_images(imagePath)
          
          augPath = img_aug_path+'/'+str(label)+'/' + im.name[:-4] + str(k) + '.jpg'
          image = tf.keras.preprocessing.image.img_to_array(image)
          plt.imsave(augPath, image/255)
          k += 1


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  

  #pred_mask = area_closing(pred_mask, area_threshold=4, connectivity=1, parent=None, tree_traverser=None)tf.keras.preprocessing.image.img_to_array

  return pred_mask[0]

def extract_image_unet(img_path, model):
  """ method used to extract the greatest region in the mask output of the 
  Unet. 
  """
  image = plt.imread(img_path)
  image = resize(image, (128,128,3), preserve_range=True)
  pred_mask = create_mask(model.predict(image[tf.newaxis, ...]/255))
  image = resize(image, (200,50,3), preserve_range=True)
    
  pred_mask = tf.keras.preprocessing.image.img_to_array(pred_mask)

   
  selem = skimage.morphology.disk(6)
  pred_mask = closing(np.squeeze(pred_mask*255), selem)
  pred_mask = opening(np.squeeze(pred_mask), selem)

  pred_mask = resize(pred_mask, (200,50, 1), preserve_range=True)
  

  try:
    labels_mask = measure.label(pred_mask) 
  except ValueError:  #raised if `y` is empty.
    print('no region found.')
    return image
   

  try:
    regions = measure.regionprops(labels_mask)
  except ValueError:  #raised if `y` is empty.
    print("no region found.")
    return image
  regions.sort(key=lambda x: x.area, reverse=True)

  if len(regions) > 1:
    for rg in regions[1:]:
        labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0


  labels_mask[labels_mask!=0] = 1
  mask = labels_mask

  skeleton = skeletonize(rescale_intensity(tf.keras.preprocessing.image.img_to_array(mask),in_range=(-1,1)))


  if mask.shape[-1] > 0:
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    img_extracted = np.where(mask, image, 255).astype(np.uint8)
    image = img_extracted/255


  skeleton = np.where(np.squeeze(skeleton) > 0)
  skeleton = np.array(skeleton)

  a = skeleton[0]
  b = skeleton[1]
 
  s = pd.Series(np.squeeze(a))
  #print(s.values)
  #print(s[s.duplicated()].index)
  duplicates = s[s.duplicated()].index
 
  #duplicate = 0
  skel_size = skeleton[0].shape[0]
  
  shifts = []
  center = int(pred_mask.shape[1]/2)
  i = 0
  dups = []
  while (i < skel_size):
  

    if (i in duplicates and not i in dups):
     
      dups.append(i)
      
      d = skeleton[1][skeleton[0] == s[i]]
     
      mid = int(np.mean(d))
    
      
   
      diff = mid-center
      shifts[-1] = -diff
    
      i += 1
      
    elif (i in duplicates and i in dups):
      continue
    else:
      diff = skeleton[1][i]-center
      
      shifts.append(-diff)
      i += 1
  
  
  k = 0
  img = image.copy()

  for diff in shifts:
    
    image[s.unique()[k],:,0] = shift(image[s.unique()[k],:,0], diff , output=None, order=3, mode='wrap', prefilter=False)
    image[s.unique()[k],:,1] = shift(image[s.unique()[k],:,1], diff , output=None, order=3, mode='wrap', prefilter=False)
    image[s.unique()[k],:,2] = shift(image[s.unique()[k],:,2], diff , output=None, order=3, mode='wrap', prefilter=False)
    
    k += 1


  for i in range (image.shape[0]):
    if (not i in skeleton[0]):
      image[i,:] = 1

  white = np.array([1, 1, 1])
  mask = np.abs(image - white).sum(axis=2) < 0.05

  # Find the bounding box of those pixels
  try:
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
  
    out = image[top_left[0]:bottom_right[0],
            top_left[1]:bottom_right[1]]
    
  except:
    return None
  
  return out


def extract_crop(ds_path, ds_final_path, input_shape, model):
  """ take the proper dataset created by augmentationImages and extract the newts
  by segmenting and cropping the pattern of interest """
   
  #!rm -rf /content/ds_final

    

  for class_name in os.listdir(ds_path):
      
      #print(class_name)
      dsPath = ds_path + '/' + class_name
      #print(dsPath)
      dsFinalPath = ds_final_path + '/' + class_name

      for img_name in os.listdir(dsPath):

        #if os.path.isfile(img_name):
        
          #print(img_name)
          imagePath = dsPath + '/' + img_name
          print(imagePath)
          augPath = dsFinalPath + '/' + img_name
          if not (os.path.isdir(dsFinalPath)):
            os.makedirs(dsFinalPath)
          
          image = extract_image_unet(imagePath, model)
          try:
          #plt.imshow(image)
          #plt.show()

            height = input_shape[0]
            width = input_shape[1]
          #print(np.max(image))
          #print(np.min(image))
  
            if (image.shape[0] < image.shape[1]):
            #image = tf.image.resize(image, [width, height])
              image = resize(image, (width, height),
                       anti_aliasing=False)
            #print("height < width")
              image = np.transpose(image,(1,0,2))
            else:
            #image = tf.image.resize(image, [height, width])
              image = resize(image, (height, width),
                       anti_aliasing=False)
        
          #image = tf.keras.preprocessing.image.img_to_array(image)
         
          #plt.imshow(image)
          #plt.show()
            plt.imsave(augPath, image)
          
          except:
            pass  



def preprocess_newts(dataset_path, img_aug_path, final_path, input_shape):
  """ take as input a folder of images, augment the images,
      extract the pattern of the newts, and straighten them automatically.
      Create the resulting dataset """
  #Handle the augmentations for each image
  augmentationImages(dataset_path, img_aug_path)
  #Handle the pattern extraction and the straightening
  Unet = tf.keras.models.load_model('my_model_Unet')
  extract_crop(img_aug_path, final_path, input_shape, Unet)

img_path = 'images'

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  print(parts[-2])
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  #img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return img

def process_path(file_path):
  #print(file_path)
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  try:
    img = decode_img(img)
    #print(np.max(img))
    print("decoded")
    counter +=1
  except:
    print("erreur décodage")

  return img, label

def format_image(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (input_shape[0], input_shape[1]))
  return image, label

def import_images(img_path, input_shape):

  AUTOTUNE = tf.data.experimental.AUTOTUNE


  ds_size = sum(len(files) for _, _, files in os.walk(img_path))
  print(ds_size)

  data_dir = pathlib.Path(img_path)

  list_ds = tf.data.Dataset.list_files(str(data_dir)+'/*/*')

  global CLASS_NAMES 
  CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

  
  #print(CLASS_NAMES)
  

  #Use Dataset.map to create a dataset of image, label pairs:

  # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
  labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
  

  

  dataset = labeled_ds.map(format_image, num_parallel_calls=AUTOTUNE)

  count_label = len(os.listdir(img_path))
  #print(count_label)

  x = np.zeros((ds_size,input_shape[0],input_shape[1],input_shape[2]))
  y = np.zeros((ds_size))


  k=0
  for image, label in dataset:
    #print(label)
    x[k,:,:,:] = image
    y[k] = np.where(label)[0][0]
    k += 1


  dataset = []

    
  #Sorting images by classes and normalize values 0=>
  for n in range(count_label):
    images_class_n = np.asarray([row for idx,row in enumerate(x) if y[idx]==n])
    dataset.append(images_class_n)

  
  print("number of different newts : "+ str(count_label))

  return dataset

#input_shape = (75,30,3)
#img_path = 'ds_final_unet'
#dataset = import_images(img_path,input_shape)

def build_network(input_shape, embeddingsize):
    '''
    Define the neural network to learn image similarity
    Input : 
            input_shape : shape of input images
            embeddingsize : vectorsize used to encode our picture   
    '''
     # Convolutional Neural Network
    network = Sequential()
    network.add(Conv2D(512, (7,7), activation='relu',
                     input_shape=input_shape,
                     kernel_initializer='he_uniform',
                     ))
    network.add(MaxPooling2D())
    network.add(Dropout(0.5))
    network.add(BatchNormalization())
    network.add(Conv2D(512, (5,5), activation='relu', kernel_initializer='he_uniform',
                     ))
    network.add(MaxPooling2D())
    network.add(Dropout(0.5))
    network.add(BatchNormalization())
    network.add(Conv2D(512, (3,3), activation='relu', kernel_initializer='he_uniform',
                     ))
    network.add(MaxPooling2D())
    #network.add(Dropout(0.5))
    
    network.add(Flatten())
    #network.add(Dense(4096, activation='relu',
     #              kernel_regularizer=l2(1e-3),
      #             kernel_initializer='he_uniform'))
    network.add(BatchNormalization())
    
    network.add(Dense(embeddingsize, activation=None,
                   kernel_initializer='he_uniform'))
    
    #Force the encoding to live on the d-dimentional hypershpere
    #network.add(Lambda(lambda x: l2_normalize(x,axis=-1)))
    
    return network

#@jit(nopython=True)
def get_batch_moderate_random(batch_size,dataset):
    """
    Create batch of APN triplets with a complete random strategy
    
    Arguments:
    batch_size -- integer 
    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c), (batch_size,w,h,c,k), (batch_size,w,h,c,k,p)
    """
    
    
    m, w, h, c = dataset[0].shape
   # print(m)
    count_label = len(os.listdir(img_path))

    P = int(batch_size*3/4)
    K = 4

    k = 0
    # initialize result
    anchors =np.zeros((P*K, w, h, c))
    positives = np.zeros((P*K, w, h, c, K))
    negatives = np.zeros((P*K, w, h, c, K, P))
    
    
    #Pick one random class for anchor
    anchor_class = np.random.choice(count_label, size=P, replace=False)

    for i in range(len(anchor_class)):

      anchor = anchor_class[i]
     
      nb_sample_available_for_class_A = dataset[anchor].shape[0]
      
      #Pick K different random pics for this class => A and P
      idxAP = np.random.choice(nb_sample_available_for_class_A,size=K+1,replace=False)
      


      negative_class = [id for id in anchor_class if id != anchor]

      for j in range(K):
        idA = idxAP[j]
        idPx = [id for id in idxAP if id != idA]
      
        anchors[i*K+j,:,:,:] = dataset[anchor][idA,:,:,:]
        
        for k in range(len(idPx)):
          idP = idPx[k]
          
          positives[i*K+j,:,:,:,k] = dataset[anchor][idP,:,:,:]

        for l in range(len(negative_class)):
            negative = negative_class[l]
            nb_sample_available_for_class_N = dataset[negative].shape[0]
            idxN = np.random.choice(nb_sample_available_for_class_N,size=K,replace=False)

            for k in range(len(idxN)):
              idN = idxN[k]
              negatives[i*K+j,:,:,:,k,l] = dataset[negative][idN,:,:,:]
          
  
    return anchors, positives, negatives

#tripletbatch = get_batch_moderate_random(batch_size=8, dataset = dataset)

def get_batch_moderate(batch_size,network,dataset):
    """
    Create batch of APN "moderate" triplets
    
    Arguments:
    draw_batch_size -- integer : number of initial randomly taken samples   
    hard_batchs_size -- interger : select the number of hardest samples to keep
    norm_batchs_size -- interger : number of random samples to add
    Returns:
    
    triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """
    X = dataset

    m, w, h,c = X[0].shape
    
    
    #Step 1 : pick a random batch to study
    studybatch = get_batch_moderate_random(batch_size,dataset)
    
    
    #Compute embeddings for anchors, positive and negatives
    A = network.predict(studybatch[0])
    P = np.zeros((A.shape[0], A.shape[1], studybatch[1].shape[4]))
    N = np.zeros((A.shape[0], A.shape[1], studybatch[1].shape[4], studybatch[2].shape[5]))  


    for i in range(studybatch[1].shape[4]):
      P[:,:,i] = network.predict(studybatch[1][:,:,:,:,i])

      for j in range(studybatch[2].shape[5]):
        N[:,:,i,j] = network.predict(studybatch[2][:,:,:,:,i,j])

    
    ### Extraction of the indices corresponding to the greatest distances
    A = tf.expand_dims(A, axis = -1)
    dist_p = tf.norm(tf.math.subtract(A,P),axis = -2)
    selectionAP = tf.math.argmax(tf.norm(A - P,axis = -2), axis = -1)

    
    sub = np.zeros((A.shape[0], A.shape[1], studybatch[1].shape[4], studybatch[2].shape[5]))    
    for i in range(N.shape[3]):
      sub[:,:,:,i] = tf.math.subtract(A,N[:,:,:,i])

    n_dist = tf.norm(sub,axis = 1)
    n_dist = tf.reduce_min(n_dist, axis = -2)
    selectionN = np.argmin(n_dist, axis = -1)
    AN = tf.norm(sub,axis = -3)


    selectionPN = np.zeros((AN.shape[0]),dtype=np.int8)
    for i in range(AN.shape[0]):
      selectionPN[i] = np.argmin(AN[i,:,selectionN[i]])
      
    
    batch_size = studybatch[2][:,0,0,0,0,0].shape[0]
  
    width = studybatch[2][0,:,0,0,0,0].shape[0]
   
    height = studybatch[2][0,0,:,0,0,0].shape[0]
   
    channels = studybatch[2][0,0,0,:,0,0].shape[0]
   
    K = studybatch[2][0,0,0,0,:,0].shape[0]
   
    N = studybatch[2][0,0,0,0,0,:].shape[0]
   

    ### Creation of the tensors containing the selected images that maximize the AP-distances and minimize the AN-distances
    positives = tf.zeros((batch_size,width,height,channels))
    negatives = tf.zeros((batch_size,width,height,channels))

    
    for i in range(studybatch[2][:,0,0,0,0,0].shape[0]):
      positives = studybatch[1][:,:,:,:,selectionAP[i]]
      negatives = studybatch[2][:,:,:,:,selectionPN[i],selectionN[i]]




    triplets = [studybatch[0][:,:,:,:], positives, negatives]

    
    return triplets


class TripletLossLayerModerate(layers.Layer):
    def __init__(self, **kwargs):
        #self.alpha = alpha
        super(TripletLossLayerModerate, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        A, P, N = inputs
        
        p_dist = tf.norm(A - P,axis = -1)
        n_dist = tf.norm(A - N,axis = -1)
        
        #tripletLoss = tf.math.reduce_sum(tf.math.add(self.alpha,tf.math.subtract(p_dist,n_dist)))

        #Using the batching formula from the paper with its soft margin variant
        tripletLoss = tf.math.reduce_sum(tf.math.log1p(tf.math.exp(tf.math.subtract(p_dist,n_dist))))
        tripletLoss = tf.dtypes.cast(tripletLoss, tf.float32)
        return(tripletLoss)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

def build_model_moderate(input_shape, network):
    '''
    Define the Keras Model for training 
        Input : 
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)
    
    '''
    
    
     # Define the tensors for the three input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input") 
    
    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    
    #TripletLoss Layer
    loss_layer = TripletLossLayerModerate(name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
    
    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)
    
    # return the model
    return network_train

def mean_average_precision(X,Y,network,rank):
    '''
    Returns
        MAP : the mean of the average precision of the model for each images of the dataset
        if rank = 1, this is the same as the best ranking score
    '''
    nb_classes = count_label
    print(nb_classes)
    m = X.shape[0]
    #nbtrain = 10
    nbevaluation = 150
    probs = np.zeros((nbevaluation))
    distances = np.zeros((nbevaluation,nb_classes))
    ypreds = np.zeros((nbevaluation,nb_classes))
    y = np.zeros((nbevaluation))

    
    #Compute all embeddings for all pics with current network
    embeddings = network.predict(X)
    
    size_embedding = embeddings.shape[1]

    ref_images = np.zeros((nb_classes,size_embedding))
    
    #generates embeddings for reference images
    for i in range(nb_classes):
          #print(dataset_train[i][:,0,0,1])
          #idx_ref = np.random.choice(dataset_train[i][:,0,0,1],size=200,replace=False)
          #print(idx_ref)
          #print(len(dataset_train[i][0]))
          #print(np.take(dataset_train[, idx_ref))
          selected_ref = dataset[i][:,:,:,:]
          #print(selected_ref.shape)
          #print(np.max(selected_ref))
          ref_images[i,:] = np.mean(network.predict(selected_ref),axis=0)
          #print(ref_images[i,:].shape)
    #print(np.mean(network.predict(selected_ref),axis=0).shape)
     
    test = compute_dist(ref_images[0,:],network.predict(np.expand_dims(dataset[15][0,:,:,:], axis=0)))
    print(test)
    for j in range(nbevaluation):
          print(j)
          for k in range(nb_classes):
              #print(X[j,:,:,:].shape)
              #print(np.max(ref_images[k,:]))
              distances[j,k] = compute_dist(ref_images[k,:],network.predict(np.expand_dims(X[j,:,:,:], axis=0)))
              #print(distances[j,k])
          #print(distances[j,:])
    #for i in range(nb_classes):
     #for k in range(nb_classes):
      #for j in range(nbevaluation):
              #print(X[j,:,:,:].shape)
       #       distances[j,k] = compute_dist(ref_images[i,:],network.predict(np.expand_dims(dataset[k][j,:,:,:], axis=0)))
        #      print(distances[j,k])
          #print(distances[j,:])
    #print("affichage des distances triées selon l'axe des classes")
    ypreds = np.argsort(distances,axis=-1)
    #print(ypreds[0])
    #print("affichage des distances triées selon l'axe des classes")
    #ypreds = np.flip(ypreds,axis = -1)
    #print(ypreds[0])
    ytrue = Y[:nbevaluation]
    #print(ytrue)

    AP = 0
    #print(len(ytrue))
    
    for i in range(len(ytrue)):
      #print(ypreds[i,:rank])
      #print(ytrue[i])
      print(ytrue)
      print(ypreds[i,0])
      for k in range(rank):
        
        if(ytrue[i] == ypreds[i,k]):
          AP += 1/(k+1)
    
    MAP = AP/len(ytrue)
    print(MAP)
    return MAP

def cumulative_matching_curve(X,Y,network,rank):
  
    '''
    Returns
        the cumulative_matching_curve metric
        if rank = 1, is the same as the best ranking score
    '''
    nb_classes = count_label
    m = X.shape[0]
    nbtrain = 1000
    nbevaluation = 50
    probs = np.zeros((nbevaluation))
    distances = np.zeros((nbevaluation,nb_classes))
    ypreds = np.zeros((nbevaluation,nb_classes))
    y = np.zeros((nbevaluation))

    
    #Compute all embeddings for all pics with current network
    embeddings = network.predict(X)
    
    size_embedding = embeddings.shape[1]

    ref_images = np.zeros((nb_classes,size_embedding))
    
    #generates embeddings for reference images
    for i in range(nb_classes):
          #print(dataset_train[i][:,0,0,1])
          #idx_ref = np.random.choice(dataset_train[i][:,0,0,1],size=200,replace=False)
          #print(idx_ref)
          #print(len(dataset_train[i][0]))
          #print(np.take(dataset_train[, idx_ref))
          selected_ref = dataset[i][:nbtrain,:,:,:]
          #print(selected_ref.shape)
          ref_images[i,:] = np.mean(network.predict(selected_ref),axis=0)
          #print(ref_images[i,:].shape)
    #print(np.mean(network.predict(selected_ref),axis=0).shape)
    
    test = compute_dist(ref_images[0,:],network.predict(np.expand_dims(dataset[15][0,:,:,:], axis=0)))
    print(test)
    for j in range(nbevaluation):
          print(j)
          for k in range(nb_classes):
              #print(X[j,:,:,:].shape)
              distances[j,k] = compute_dist(ref_images[k,:],network.predict(np.expand_dims(X[j,:,:,:]/255, axis=0)))
              #print(distances[j,k])
          #print(distances[j,:])
    #for i in range(nb_classes):
     #for k in range(nb_classes):
      #for j in range(nbevaluation):
              #print(X[j,:,:,:].shape)
       #       distances[j,k] = compute_dist(ref_images[i,:],network.predict(np.expand_dims(dataset[k][j,:,:,:], axis=0)))
        #      print(distances[j,k])
          #print(distances[j,:])
    #print("affichage des distances triées selon l'axe des classes")
    ypreds = np.argsort(distances,axis=-1)
    #print(ypreds[0])
    #print("affichage des distances triées selon l'axe des classes")
    #ypreds = np.flip(ypreds,axis = -1)
    #print(ypreds[0])
    ytrue = Y[:nbevaluation]
    #print(ytrue)

    present = 0
    #print(len(ytrue))
    
    for i in range(len(ytrue)):
      #print(ypreds[i,:rank])
      #print(ytrue[i])
      for k in range(rank):
        #print(ytrue)
        #print(ypreds[i,k])
        if(ytrue[i] == ypreds[i,k]):
          present += 1
    
    CMC = present/len(ytrue)
    print(CMC)
    return CMC

def compute_interdist(network):
    '''
    Computes sum of distances between all classes embeddings on our reference test image: 
        d(0,1) + d(0,2) + ... + d(0,9) + d(1,2) + d(1,3) + ... d(8,9)
        A good model should have a large distance between all theses embeddings
        
    Returns:
        array of shape (nb_classes,nb_classes) 
    '''
    nb_classes = count_label
    res = np.zeros((nb_classes,nb_classes))
    
    ref_images = np.zeros((nb_classes,28,28,3))
    
    #generates embeddings for reference images
    for i in range(nb_classes):
        ref_images[i,:,:,:] = dataset[i][0,:,:,:]
    ref_embeddings = network.predict(ref_images)
    
    for i in range(nb_classes):
        for j in range(nb_classes):
            res[i,j] = compute_dist(ref_embeddings[i],ref_embeddings[j])
    return res

def compute_dist(a,b):
    return np.sum(np.square(a-b))

#def predict_batch(img_path):

def train_triplet_loss(embedding_size, input_shape, dataset):
  import time
  #embedding_size = 128
  tf.keras.backend.clear_session()
  model = build_network(input_shape,embedding_size)
  model_trained = build_model_moderate(input_shape, model)

  model_trained.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))

  n_iter = 3000
  t_stop = 0
  iter_stop = 0


  if (iter_stop != 0):
    n_iteration = iter_stop
  else:
    n_iteration = 0

  evaluate_every = 1
  n_val = 1000
  print("Starting training process!")
  print("-------------------------------------")

  t_start = time.time()

  if (t_stop != 0):
    t_start = t_stop


  for i in range(1, n_iter+1):
      triplets = get_batch_moderate(14,model,dataset)
      #triplets = get_batch_hard(32,16,16,model)
    
      #print(triplets.shape)
      loss = model_trained.train_on_batch(triplets, None)
      n_iteration += 1
      t_stop = time.time()
      iter_stop = n_iteration

      if i % evaluate_every == 0:
          print("\n ------------- \n")
          print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time()-t_start)/60.0,np.nan_to_num(loss),n_iteration))
        
      
  return model

#final_path = "ds_unet"
#input_shape = (75,30,3)
#dataset = import_images(final_path,input_shape)

#model = train_triplet_loss(128, input_shape,dataset)
def process_path_new(file_path):
  #print(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  try:
    img = decode_img(img)
    #print(np.max(img))
    #print("decoded")
    counter +=1
  except:
    pass

  return img, file_path

def format_image_new(image, file_path):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (input_shape[0], input_shape[1]))
  return image, file_path


def distinguishNewNewts(dist, ids, threshold_new, x_new):


  
  

  distances = np.zeros((len(ids),len(ids)))

  print(distances)

  n = distances.shape[0]

  for i in range(n):
    for j in range(n):
      distances[i,j] = compute_dist(model.predict(np.expand_dims(x_new[ids[i],:,:,:], axis=0)), 
                                    model.predict(np.expand_dims(x_new[ids[j],:,:,:], axis=0)))
  
  mask = [[1 if x>y else 0 for x in range(n)] for y in range(n)]

  mask = np.array(mask)

  print(mask)

  distances = distances*mask

  print(distances)

  

  distances = np.where((distances < threshold_new) & (distances > 0))

  print(distances)

  same_newts = []
  for i in range(len(distances[0])):
    same_newts.append((ids[distances[0][i]],ids[distances[1][i]]))

  print(same_newts)


  return same_newts

  #id = names.loc[im.name]
  #print(id)
  #label = labels.iloc[id]
  #print('label : '+label)
  #id_label = np.where(labels_uniques == label)
  #print(id_label)
  #imagePath = basepath + '/' + im.name


def addToDataset(img_path, new_path, model, unet, input_shape, saveEmbeddings = False):
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  
  #data = pd.read_csv('dataset.csv')

  #labels = data.iloc[:,1]
  shutil.rmtree('images_aug_path')
  shutil.rmtree('images_cropped_path')
  shutil.rmtree('images')
  #names = data.iloc[:,0]
  os.mkdir('images_aug_path')
  os.mkdir('images_cropped_path')
  os.mkdir('images')
  image_aug_path = 'images_aug_path'
  image_cropped_path = 'images_cropped_path'

  for img in os.listdir(new_path):
    image_path = new_path + '/' + img
    image = preprocess_images(image_path)
    print(np.max(image))
    image_path_aug = image_aug_path + '/' + img 
    plt.imsave(image_path_aug, image/255)
    imageCropped = extract_image_unet(image_path_aug, unet)
    print(np.max(imageCropped))
    image_path_cropped = image_cropped_path + '/' + img 
    plt.imsave(image_path_cropped, imageCropped)

  new_path = image_cropped_path

  ds_size = sum(len(files) for _, _, files in os.walk(img_path))
  new_ds_size = sum(len(files) for _, _, files in os.walk(new_path))
  print(ds_size)
  print(new_ds_size)

  data_dir = pathlib.Path(img_path)
  new_data_dir = pathlib.Path(new_path)

  global CLASS_NAMES
  CLASS_NAMES = np.array([folder for folder in os.listdir(img_path)])
  print(CLASS_NAMES)

  new_img_names = np.array([folder for folder in os.listdir(new_path)])
  

  list_ds = tf.data.Dataset.list_files(str(data_dir)+'/*/*', shuffle=False)
  new_list_ds = tf.data.Dataset.list_files(str(new_data_dir)+'/*', shuffle=False)

  for img in new_list_ds:
    print(img)

  labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

  dataset = labeled_ds.map(format_image, num_parallel_calls=AUTOTUNE)

  new_ds = new_list_ds.map(process_path_new, num_parallel_calls=AUTOTUNE) 

  new_dataset = new_ds.map(format_image_new, num_parallel_calls=AUTOTUNE)

  count_label = len(os.listdir(img_path))
  #print(count_label)

  x = np.zeros((ds_size,input_shape[0],input_shape[1],input_shape[2]))
  y = np.zeros((ds_size))
  x_new = np.zeros((new_ds_size,input_shape[0],input_shape[1],input_shape[2]))
  file_paths = []

  k = 0
  for image, label in dataset:
    #print(label)
    x[k,:,:,:] = image
    #print(image)
    y[k] = np.where(label)[0][0]
    k += 1

  k = 0
  for image, file_path in new_dataset:
    x_new[k,:,:,:] = image
    file_paths.append(file_path)
    #print(file_path)
    k += 1

  dataset = []

    
  #Sorting images by classes and normalize values 0=>
  for n in range(count_label):
    images_class_n = np.asarray([row for idx,row in enumerate(x) if y[idx]==n])
    dataset.append(images_class_n)


 
  #Compute the center of each class in the embedding space
  nb_classes = count_label
  print(nb_classes)
  probs = np.zeros((new_ds_size))
  distances = np.zeros((new_ds_size,nb_classes))
  ypreds = np.zeros((new_ds_size,nb_classes))
  y = np.zeros((new_ds_size))

    
  #Compute the embedding of every image 
  embeddings = model.predict(x)
    
  size_embedding = embeddings.shape[1]

  ref_images = np.zeros((nb_classes,size_embedding))
    
  #generates embeddings for reference images
  for i in range(nb_classes):

      selected_ref = dataset[i][:,:,:,:]
      ref_images[i,:] = np.mean(model.predict(selected_ref),axis=0)

  

  for i in range(new_ds_size):

    for k in range(nb_classes):

      distances[i,k] = compute_dist(ref_images[k,:],model.predict(np.expand_dims(x_new[i,:,:,:], axis=0)))


  ypreds = np.argsort(distances,axis=-1)
  print(distances)
  distances_sorted = np.sort(distances, axis=-1)

  min_dist = distances_sorted[:,0]

  print(min_dist)
  
  y_new = ypreds[:,0]

  print(y_new)

  
  
  class_identified = CLASS_NAMES[y_new]

  print(class_identified)

  #dist_test = compute_dist(ref_images[13,:],model.predict(np.expand_dims(x_new[0,:,:,:], axis=0)))
  #plt.imshow(x_new[0,:,:,:])
  #plt.show()
  #print(dist_test)
  threshold = 300
  ids = np.arange(new_ds_size)
  ids_new = np.where(min_dist > threshold)[0]
  ids_old = [id for id in ids if id not in ids_new]
  print(ids)

  threshold_new = 400

  same_newts_ids = distinguishNewNewts(min_dist, ids_new, threshold_new, x_new)

  ids_to_remove = []

  for i in range(len(same_newts_ids)):
    if (same_newts_ids[i][1] not in ids_to_remove):
      ids_to_remove.append(same_newts_ids[i][1])
  

  print(ids_to_remove)

  ids_to_keep = []

  for i in range(len(same_newts_ids)):
    if ((same_newts_ids[i][0] not in ids_to_remove) and (same_newts_ids[i][0] not in ids_to_keep)):
      ids_to_keep.append(same_newts_ids[i][0])
  
  data = pd.read_csv('data.csv')
  labels_unique = np.unique(data.iloc[:,1])
  if (labels_unique.size == 0):
    labels_unique = 0
  for i in range(len(ids_to_keep)):
    y_new[ids_to_keep[i]] = np.max(labels_unique) + i + 1

  
  ids_to_keep = ids_to_keep + ids_old
  y_new_kept = y_new[ids_to_keep]

  print(ids_to_keep)

  x_new_kept = x_new[ids_to_keep,:,:,:]

  
  
  file_paths_kept = []

  print(len(file_paths))
  for i in range(new_ds_size):
    if (i not in ids_to_remove):
      
      file_path = tf.strings.split(file_paths[i], os.path.sep)
      file_name = file_path[-1].numpy().decode("utf-8")
      print(file_name)
      file_paths_kept.append(file_name)
  #print(f_kept)
  print(file_paths_kept)
  #for k in range(x_new_kept.shape[0]):
  x_new_kept = ((x_new_kept+1)*127.5)/255

  print(x_new_kept.shape)
  print(np.min(x_new_kept))

  
  temp_folder = 'images'
  #os.mkdir(temp_folder)
  
  rows_list = []
  #data.head()
  for i in range(x_new_kept.shape[0]):
    plt.imsave(temp_folder+ '/'+ str(file_paths_kept[i]), x_new_kept[i,:,:,:])
    dict1 = {
            "file_name": str(file_paths_kept[i]),
             "label": y_new_kept[i]
        }
    rows_list.append(dict1)

  df = pd.DataFrame(rows_list, columns=['file_name', 'label'])
  print(data.shape)
  data = data.append(df)
  #data.tail()
  print(data.shape)
  data.to_csv (r'data.csv', index = False, header=True)

  #os.remove("demofile.txt") 

  #preprocess_newts(img_path,aug_path,final_path,input_shape)

  print("number of different newts : "+ str(count_label))

  
unet = load_model('my_model_Unet')
model = load_model('saved_cnn')
img_path = 'ds_final_unet_lite/content/ds_final/train'
new_path = 'newt_images'
input_shape = (75,30,3)
#threshold = 300, threshold_new = 400
addToDataset(img_path, new_path, model, unet, input_shape)




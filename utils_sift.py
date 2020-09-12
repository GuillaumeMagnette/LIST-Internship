import numpy as np
import cv2
import pysift
from matplotlib import pyplot as plt
import logging
import tensorflow as tf
import os
import pathlib
import shutil
import pandas as pd
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.models import load_model, save_model

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

from random import random, uniform


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


def augmentationImages(img_path,img_aug_path):

  """ Create a proper dataset where each folder contains the images of one 
  specimen, randomly augmented 20 times for each "real" image present in the 
  original dataset """

  #!rm -rf /content/databaseAug #useful when running the method again for makedirs

  img_list = os.listdir(img_path)

  for img_name in img_list:
      os.makedirs(img_aug_path + '/' + str(img_name[:-4]))


  print("Creating the augmented dataset...")
  # List all files in a directory using scandir()
  basepath = img_path
  with os.scandir(basepath) as images:
    for im in images:

        imagePath = basepath + '/' + im.name
        print(imagePath)
        k = 0

        while (k < 20):  
          image = preprocess_images(imagePath)
          
          augPath = img_aug_path+'/'+ im.name[:-4] +'/' + im.name[:-4] + str(k) + '.jpg'
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
          #print(imagePath)
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

  if(os.path.exists(img_aug_path)):
    shutil.rmtree(img_aug_path)
  if(os.path.exists(final_path)):
    shutil.rmtree(final_path)
  #Handle the augmentations for each image
  newDataset = os.listdir(dataset_path)
  new_path = dataset_path + '/' + newDataset[0]
  for area_name in os.listdir(new_path):
    augmentationImages(new_path + '/' + area_name, img_aug_path + '/' + area_name)
  #Handle the pattern extraction and the straightening
  Unet = tf.keras.models.load_model('unet')
  #print(Unet.summary())
  for area_name in os.listdir(img_aug_path):
    extract_crop(img_aug_path + '/' + area_name, final_path + '/' + area_name, input_shape, Unet)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))




def compareTwoNewts(dir_path1,dir_path2,nbr_comparisons):

  img_dir1 = os.listdir(dir_path1)
  img_dir2 = os.listdir(dir_path2)
  img_dir1 = img_dir1[:nbr_comparisons]
  img_dir2 = img_dir2[:nbr_comparisons]
  goods = []
  
  for im_name1 in img_dir1:
    
    for im_name2 in img_dir2:
      #print(i)
      #print(j)
      
        img_path1 = dir_path1 + '/' + im_name1
        img_path2 = dir_path2 + '/' + im_name2
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        #ret1,thresh1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
        th1 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
              cv2.THRESH_BINARY,11,2)
        #ret2,thresh2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
              cv2.THRESH_BINARY,11,2)
      
        kp1, des1 = pysift.computeKeypointsAndDescriptors(th1)
        kp2, des2 = pysift.computeKeypointsAndDescriptors(th2)

        # Initialize and use FLANN
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for m, n in matches:
          if m.distance < 0.7 * n.distance:
            good.append(m)
        goods.append(len(good))
      
  averageSimilarities = np.mean(goods)
  print(averageSimilarities)
  if (averageSimilarities > 1):
    sameNewt = True
  else:
    sameNewt = False  
  
  return sameNewt

#img_dir1 = 'cropDataset\Bascha_P01_T01_K04_F_Adult_4240_20190330204648'
#img_dir2 = 'cropDataset\Bascha_P01_T01_K04_M_Adult_4230_20190330200044'

#is_same_newt = compareTwoNewts(img_dir1, img_dir2,5)
#print(is_same_newt)

def regroupSameNewtsInit(cropped_path):
  
  diff_newts_ids = []
  same_newts_ids = []
  i = 0
  list_dir = os.listdir(cropped_path)

  for class_path1 in list_dir:
    j = 0
    for class_path2 in list_dir:
      if (j > i):
        img_path1 = cropped_path + '/' + class_path1
        img_path2 = cropped_path + '/' + class_path2
        if compareTwoNewts(img_path1,img_path2,1):
          same_newts_ids.append((i,j))
        else:
          diff_newts_ids.append((i,j))
      j += 1
    i += 1

  ids_to_remove = []

  for i in range(len(same_newts_ids)):
    if (same_newts_ids[i][1] not in ids_to_remove):
      ids_to_remove.append(same_newts_ids[i][1])
    
  print(ids_to_remove)

  for id in ids_to_remove:
    shutil.rmtree(cropped_path + '/' + list_dir[id] )
  

  new_list_dir = os.listdir(cropped_path)
  esti_nbr_newts = len(new_list_dir)

  return esti_nbr_newts


def regroupSimilarAreas(new_cropped_path):

  #nbr_new_diff_newts = regroupSameNewtsInit(new_cropped_path)

  diff_newts_ids = []
  same_newts_ids = []
  i = 0
  
  list_dir = os.listdir(new_cropped_path)
  #list_dir2 = os.listdir(cropped_path2)

  for class_path1 in list_dir:
    j = 0
    for class_path2 in list_dir:
      if(j > i):
        img_path1 = new_cropped_path + '/' + class_path1
        img_path2 = new_cropped_path + '/' + class_path2
        if compareTwoNewts(img_path1,img_path2,3):
          same_newts_ids.append((i,j))
        else:
          diff_newts_ids.append((i,j))
      j += 1
    i += 1

  ids_to_remove = []

  for i in range(len(same_newts_ids)):
    if (same_newts_ids[i][1] not in ids_to_remove):
      ids_to_remove.append(same_newts_ids[i][1])
    
  print(ids_to_remove)
  

  for id in ids_to_remove:
    shutil.rmtree(new_cropped_path + '/' + list_dir[id] )

  new_list_dir = os.listdir(new_cropped_path)
  esti_nbr_newts = len(new_list_dir)

  return esti_nbr_newts
  
#nbr_newts = regroupSameNewts('cropDataset')

def regroupSameNewts(cropped_path,new_cropped_path):

  nbr_new_diff_newts = regroupSameNewtsInit(new_cropped_path)

  diff_newts_ids = []
  same_newts_ids = []
  i = 0
  list_dir = os.listdir(cropped_path)
  list_dir2 = os.listdir(new_cropped_path)

  for class_path1 in list_dir:
    j = 0
    for class_path2 in list_dir2:
  
      img_path1 = cropped_path + '/' + class_path1
      img_path2 = new_cropped_path + '/' + class_path2
      if compareTwoNewts(img_path1,img_path2,1):
        same_newts_ids.append((i,j))
      else:
        diff_newts_ids.append((i,j))
      j += 1
    i += 1

  ids_to_remove = []

  for i in range(len(same_newts_ids)):
    if (same_newts_ids[i][1] not in ids_to_remove):
      ids_to_remove.append(same_newts_ids[i][1])
    
  print(ids_to_remove)

  for id in ids_to_remove:
    shutil.rmtree(new_cropped_path + '/' + list_dir2[id] )

  new_list_dir = os.listdir(new_cropped_path)
  for i in range(len(new_list_dir)):
    shutil.move(new_cropped_path + '/' + new_list_dir[i], cropped_path)

  new_list_dir = os.listdir(cropped_path)
  esti_total_nbr_newts = len(new_list_dir)

  return esti_total_nbr_newts

#def scanAreasNewts(cropped_path)

def create_area(img_path, cropped_path):
  img_path = img_path + '/' + os.listdir(img_path)[0]
  list_area = []
  for img_name in os.listdir(img_path):
    area = img_name.split('_')[0]
    if (area not in os.listdir(cropped_path)):
      os.makedirs(cropped_path + '/' + area)
    if (area not in list_area):
      list_area.append(area)

  list_img = os.listdir(img_path)
  for area in list_area:
    res = [idx for idx in list_img if idx.split('_')[0].lower() == area.lower()]
    os.makedirs(img_path + '/' + area)
    for newt_from_area in res:
      print(newt_from_area)
      shutil.move(img_path + '/' + newt_from_area, img_path + '/' + area)




if __name__ == '__main__':
  img_path = 'baseDataset'
  new_base_path = 'newDataset'
  aug_path = 'augDataset'
  new_aug_path = 'newAugDataset'
  cropped_path = 'cropDataset'
  new_cropped_path = 'newCropDataset'

  input_shape = (75,30,3)

  preprocess_newts(img_path,aug_path,cropped_path,input_shape)
  preprocess_newts(new_base_path,new_aug_path,new_cropped_path,input_shape)


  nbr_newts = regroupSameNewts('cropDataset', 'newCropDataset')
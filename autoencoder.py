
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score, silhouette_samples
import seaborn as sns


import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from ax import optimize

import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import regularizers
import umap
import hdbscan
import numpy as np


#width = 56
#height = 56

train_path = '/content/newtDatasetLite/train'
train_path = '/content/finalDataset'
#train_path = '/content/newtDataset/train'

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
  image = tf.image.rgb_to_grayscale(image)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (input_shape[0], input_shape[1]))
  return image, label



def import_images_AE(img_path, input_shape):

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

  
  print("number of different newts : "+ str(count_label))

  return x,y

x,y = import_images_AE()
x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])) 


#x_train = x_train_lite
#x_test = x_test_lite

def autoencoder(dims, act='relu'):
    n_stacks = len(dims) - 1
    x = Input(shape=(dims[0],), name='input')
    h = x
    for i in range(n_stacks - 1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)
    for i in range(n_stacks - 1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)
    h = Dense(dims[0], name='decoder_0')(h)

    return Model(inputs=x, outputs=h)

shape = [x_train.shape[-1], 2000, 2000, 4000, 2048]
autoencoder = autoencoder(shape)

hidden = autoencoder.get_layer(name='encoder_%d' % (len(shape) - 2)).output
encoder = Model(inputs=autoencoder.input, outputs=hidden)

def train_autoencoder():
    autoencoder.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005))
    autoencoder.fit(
            x,
            x,
            batch_size=100,
            epochs=200,
            #validation_data=(x_test, x_test),
            verbose=1)

from sklearn.preprocessing import StandardScaler
embedding = encoder.predict(x)
embedding = StandardScaler().fit_transform(embedding)

def hdbscan_evaluation_function(parameterization):
    x = []
    x.append(parameterization.get(f"min_samples"))
    x.append(parameterization.get(f"min_cluster_size"))
    x.append(parameterization.get(f"n_neighbors"))
    x.append(parameterization.get(f"min_dist"))
    x.append(parameterization.get(f"n_components"))
    x = np.array(x)


    umap_embedding = umap.UMAP(
    n_neighbors=x[2],
    min_dist=x[3]/100,
    n_components=x[4],
    random_state=42,
    ).fit_transform(embedding)

    print(x)
    print(x.shape)
    print(int(x[0]))
    print(int(x[1]))
    labels = hdbscan.HDBSCAN(
    min_samples=int(x[0]),
    min_cluster_size=int(x[1]),
    ).fit_predict(umap_embedding)

    if (len(np.unique) > 2):
      scores = -(davies_bouldin_score(umap_embedding, labels))  + silhouette_score(umap_embedding, labels)
      print(scores)
    else:
      scores = -5
    #silhouette_samples(umap_embedding, labels, metric='euclidean'), + calinski_harabasz_score(umap_embedding, labels)
    
    # In our case, standard error is 0, since we are computing a synthetic function.
    return scores

def optimize_parameters():

    best_parameters, best_values, experiment, model = optimize(
        parameters=[
          {
            "name": "n_neighbors",
            "type": "range",
            "bounds": [10, 500],
          },
          {
            "name": "min_dist",
            "type": "range",
            "bounds": [0, 100],
          },
          {
            "name": "n_components",
            "type": "range",
            "bounds": [2,6],
          },
          {
            "name": "min_samples",
            "type": "range",
            "bounds": [10, 500],
          },
          {
            "name": "min_cluster_size",
            "type": "range",
            "bounds": [10, 500],
          },
        ],
        # Booth function
        experiment_name="test",
        objective_name="scores",
        evaluation_function=hdbscan_evaluation_function,
        #evaluation_function=lambda p: (p["x1"] + 2*p["x2"] - 7)**2 + (2*p["x1"] + p["x2"] - 5)**2,
        total_trials=20,
        minimize=False,
    )

    return best_parameters, best_values, experiment, model

best_parameters,_,_,_ = optimize_parameters()

def apply_umap():

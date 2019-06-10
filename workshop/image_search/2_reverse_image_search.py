import os
import time
import random
import pickle
import argparse
import keras
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from scipy.spatial import distance
import matplotlib.pyplot as plt
from PIL import ImageFile, Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query', required=False, default='../images/dognugget.png', help="The image to do an query on")
parser.add_argument('-f', '--features', required=False, default='../data/features_caltech256.p', help="The location of the trained features")
parser.add_argument('-g', '--graph', action='store_true', help="Set to false if you want to output an image instead of a graph")
args = vars(parser.parse_args())

"""
Using the extracted features, pca_features and pca from 1_extract_features.py
to query an reverse image search
"""

def load_image(path):
    """
    Pre-processing an image file to feature vectors in order to serve as
    input for the network
    """
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

def get_closest_images(query_image_idx, num_results=5):
    """
    Take the result images and combine them into a single image for easy display.
    """
    distances = [ distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest

def get_concatenated_images(indexes, thumb_height):
    """
    Compute the cosine distance between the PCA features of the query image in our dataset.
    """
    thumbs = []
    for idx in indexes:
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

# pre-trained VGG16 network with imagenet weights
model = keras.applications.VGG16(weights='imagenet', include_top=True)
# Remove the last prediction layer in favour of the fc2 layer, giving
# accurate representations of the image
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

# load the extracted features
images, pca_features, pca = pickle.load(open(args["features"], 'rb'))

# load query image and extract features
new_image, x = load_image(args["query"])
new_features = feat_extractor.predict(x)

# project it into pca space
new_pca_features = pca.transform(new_features)[0]

# calculate its distance to all the other images pca feature vectors
distances = [ distance.cosine(new_pca_features, feat) for feat in pca_features ]
idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:5]  # grab first 5
results_image = get_concatenated_images(idx_closest, 200)

if args["graph"]:
    print('[INFO] Displaying graph')
    # display the query image
    plt.figure(figsize = (5,5))
    plt.imshow(new_image)

    # display the resulting images
    plt.figure(figsize = (16,12))
    plt.imshow(results_image)
    plt.show()
else:
    # creating a timestamp
    timestr = time.strftime('%Y-%m-%d at %H.%M.%S')
    print('[INFO] Saving output to file at: output/' + timestr + '_output.jpg')
    # save the result image(s)
    im = Image.fromarray(results_image)
    im.save("output/" + timestr + "_output.jpg") # save the results

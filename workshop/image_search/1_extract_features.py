import os
import time
import random
import pickle
import argparse
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import ImageFile, Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=False, default='Caltech101', help="location of the dataset")
parser.add_argument('-s', '--save', required=False, default='features_calltech101', help="save location of the trained features")
args = vars(parser.parse_args())

"""
Extracting features from a given dataset using a previously trained VGG16
network
"""
# truncated files fix
ImageFile.LOAD_TRUNCATED_IMAGES = True

images_path = '../datasets/' + args["dataset"]
image_extensions = ['.jpg', '.png', '.jpeg']
max_num_images = 50000
tic = time.perf_counter()
features = []

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

# pre-trained VGG16 network with imagenet weights
model = keras.applications.VGG16(weights='imagenet', include_top=True)
# Remove the last prediction layer in favour of the fc2 layer, giving
# accurate representations of the image
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

# grab images in the image_path folder and add it to the images array
# max_num_images is effective here
images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for
          f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)),
                                                      max_num_images))]

# loop over every image in the images array, extract its features and append
# it to the features list
for i, image_path in enumerate(images):
    if i % 10 == 0:
        toc = time.perf_counter()
        elap = toc-tic
        print("[INFO] Analyzing image %d / %d – Time: %4.4f seconds" % (i, len(images), elap))
        TIC = time.perf_counter()
    img, x = load_image(image_path)
    feat = feat_extractor.predict(x)[0]
    features.append(feat)

print('[INFO] Finished extracting features for %d images' % len(images))


# reduce the dimensionality of the feature vectors down to 300 to speed things
# up.
features = np.array(features)
pca = PCA(n_components=300)
pca.fit(features)
pca_features = pca.transform(features)

# save extracted images, pca_features and pca to use later
pickle.dump([images, pca_features, pca], open('../data/' + args["save"] + '.p', 'wb'))

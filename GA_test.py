import numpy as np
import skimage.io, skimage.color, skimage.feature
import os
import pickle

test_fruits = ["Test Apple Braeburn","Test Lemon Meyer","Test Mango","Test Raspberry"]
test_inputs = np.zeros(shape=(662, 360))
test_outputs = np.zeros(shape=(662))

index = 0
label = 0

for fruit in test_fruits:
    current_directory = os.path.join(os.path.sep, fruit)
    all_images = os.listdir(os.getcwd()+current_directory)
    for image in all_images:
        if image.endswith(".jpg"):
            fruit_feature = skimage.io.imread(fname = os.path.sep.join([os.getcwd(),current_directory,image]), as_gray = False)
            fruit_feature_hsv = skimage.color.rgb2hsv(rgb = fruit_feature)
            hist = np.histogram(a=fruit_feature_hsv[:,:,0],bins = 360)
            test_inputs[index,:] = hist[0]
            test_outputs[index] = label
            index = index + 1
    label = label +1

pickle.dump(test_inputs, open("test_inputs.pkl", "wb"))
pickle.dump(test_outputs, open("test_outputs.pkl", "wb"))

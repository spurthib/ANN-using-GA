import numpy as np
import skimage.io, skimage.color, skimage.feature
import pickle
import os

fruits = ["Apple Braeburn","Lemon Meyer","Mango","Raspberry"]
#all images in above four classes equals 1962
#360 is number of bins in which each image is represented
inputs = np.zeros(shape=(1962,360))
outputs = np.zeros(shape=(1962))

index = 0
label = 0

for fruit in fruits:
    #giving the path where the data set exists
    current_directory = os.path.join(os.path.sep, fruit) 
    #reads each folder mentioned in the list above from the path for all images
    all_images = os.listdir(os.getcwd()+current_directory)
    for image in all_images:
        if image.endswith(".jpg"):
            fruit_feature = skimage.io.imread(fname = os.path.sep.join([os.getcwd(),current_directory,image]), as_gray = False)
            #converts rgb to hsv(hue channel) to isolate color information for easy calculations
            fruit_feature_hsv = skimage.color.rgb2hsv(rgb = fruit_feature)
            #we represnt hue channel in histogram format of 360 bins to reduce the amount of data
            hist = np.histogram(a=fruit_feature_hsv[:,:,0],bins = 360)
            inputs[index,:] = hist[0]
            outputs[index] = label
            index = index + 1
    label = label +1
#adding the features of all images in the form 1962 X 360 to inputs pickle file
pickle.dump(inputs, open("inputs.pkl","wb"))
#adding the classlabel[0,1,2,3] for which each image belongs to the outputs pickle file
#outputs pickle file has the format 1962, each for one image
pickle.dump(outputs, open("outputs.pkl","wb"))

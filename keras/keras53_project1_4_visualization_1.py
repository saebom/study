# import Numpy, Scipy, and Matplotlib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# import Keras's functional api
from keras.models import Model, Sequential

model = Sequential()

# get the weights from the last layer
gap_weights = model.layers[-1].get_weights()[0]

# create a new model to output the feature maps and the predicted labels
cam_model = Model(inputs=model.input, 
                    outputs=(model.layers[-3].output, model.layers[-1].output)) 

# make the prediction for a set of test images
features, results = cam_model.predict(test_img)

# check the prediction for 10 test images
for idx in range(10):   
    # get the feature map of the test image
    features_for_one_img = features[idx, :, :, :]

    # map the feature map to the original size
    height_roomout = train_img_size_h / features_for_one_img.shape[0]
    width_roomout = train_img_size_w / features_for_one_img.shape[1]
    cam_features = sp.ndimage.zoom(features_for_one_img, (height_roomout, width_roomout, 1), order=2)
        
    # get the predicted label with the maximum probability
    pred = np.argmax(results[idx])
    
    # prepare the final display
    plt.figure(facecolor='white')
    
    # get the weights of class activation map
    cam_weights = gap_weights[:, pred]

    # create the class activation map
    cam_output = np.dot(cam_features, cam_weights)
    
    # draw the class activation map
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    buf = 'Predicted Class = ' + fashion_name[pred] + ', Probability = ' + str(results[idx][pred])
    plt.xlabel(buf)
    plt.imshow(t_pic[idx], alpha=0.5)
    plt.imshow(cam_output, cmap='jet', alpha=0.5)
     
    plt.show()  
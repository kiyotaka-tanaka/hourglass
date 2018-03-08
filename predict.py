import tensorflow as tf
from model import model
from dataloader import youtube,read_image

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--image_path",type=str,help="image path",default="./image.jpg")
parser.add_argument("--model_path",type=str,help="restore model path ",default="./models/1200.ckpt")
args = parser.parse_args()

dataloader = youtube(folder="./GT_frames",mat_file="YouTube_Pose_dataset.mat",batch_size=10)

model_ = model(dropout_rate=0.1,learning_rate=0.001,dataloader=dataloader)



model_.restore(args.model_path)

image = read_image(args.image_path)
image = np.expand_dims(image,axis=0)
output = model_.sess.run(model_.sigmoid_out,feed_dict={model_.input_image:image})[0]

head = output[:,:,0:1]

head = np.squeeze(head)

plt.imshow(head,cmap="hot",interpolation="nearest")
plt.show()

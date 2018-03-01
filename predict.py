import tensorflow as tf
from model import model
from dataloader import youtube,read_image

import cv2
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--image_path",type=str,help="image path",default="./image.jpg")
args = parser.parse_args()

dataloader = youtube(folder="./GT_frames",mat_file="YouTube_Pose_dataset.mat",batch_size=10)

model_ = model(dropout_rate=0.1,learning_rate=0.001,dataloader=dataloader)



model_.restore("./models/1200.ckpt")

image = read_image(args.image_path)
image = np.expand_dims(image,axis=0)
output = model_.sess.run(model_.output,feed_dict={model_.input_image:image})

print (output[0][0])


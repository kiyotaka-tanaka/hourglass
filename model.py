import tensorflow as tf
import numpy as np




class hourglass:
    def __init__(self,stack_number=4):

        """
        Implementation of single hourglass network
        
        """
        self.stack = stack_number
        self.input_tensor = tf.placeholder(tf.float32,shape=[None,256,256,3])
        # TODO how to preprocess data 
        #self.out_tensor = tf.placeholder(tf.float32,size=[None])

        
    def generate_network(self,input_tensor,name="hourglass"):
        with tf.variable_scope(name):
            # 256x256x3 -> 128x1128x256 
            l1 = tf.layers.conv2d(input_tensor,filters=256,kernel_size=(7,7),strides=(2,2),padding="SAME")
            l1 = relu_batch_norm(l1)
            #128x128x256 -> 64x64x256
            l2 = tf.layers.MaxPooling2D(out,pool_size=(2,2),strides=(1,1),padding="same")
            
            l2 = self.residual(l2,name="res1")
            l2 = relu_batch_norm(l2)
            #64x64x256 -> 32x32x256
            l3 = tf.layers.MaxPooling2D(out,pool_size=(2,2),strides=(1,1),padding="SAME")
            l3 = self.residual(l3,name="res2")
            l3 = relu_batch_norm(l3)
            # 32x32x256 -> 16x16x256
            l4 = tf.layers.MaxPooling2D(l4,pool_size=(2,2),strides=(1,1),padding="SAME")
			l4 = self.residual(l4,name="res3")            
            l4 = relu_batch_norm(l4)

            #16x16x256 -> 8x8x256
            l5 = tf.layers.MaxPooling2D(l5,pool_size=(2,2),strides=(1,1),padding="SAME")
            l5 = self.residual(l5,name="res4")
            l5 = relu_batch_norm(l5)

           	
            #8x8x256 -> 4x4x256
            l6 = tf.layers.MaxPooling2D(l6,pool_size=(2,2),strides=(1,1),padding="SAME")
            l6 = self.residual(l6)
            l6 = relu_batch_norm(l6)

            up1 = tf.image.resize_nearest_neigbors(l6,size=[16,16])

    def residual(self,input_tensor,name="residual"):
        """
        residual block same as Stacked hourglass network paper
        """
        with tf.variable_scope(name):
            
            out = tf.layers.conv2d(input_tensor,filters=128,kernel_size=(1,1),strides=(1,1)padding="SAME")
            out = relu_batch_norm(out)
            out = tf.layers.conv2d(out,filters=128,kernel_size=(3,3),strides=(1,1),padding="SAME")
            out = relu_batch_norm(out)
            out = tf.layers.conv2d(out,filters=256,kernel_size=(1,1),strides=(1,1),padding="SAME")
            
        return tf.add(input_tensor,out)

    
    
    
	    
    


### helper functions ####
def relu_batch_norm(x):
    return tf.nn.relu(tf.layers.batch_normalization(x))

import tensorflow as tf
import numpy as np
from dataloader import youtube


maxpool = tf.layers.MaxPooling2D(pool_size=[2,2],strides=(2,2),padding="SAME")

class model:
    def __init__(self,dropout_rate,learning_rate,dataloader,is_training=True):
        
        """
        Implementation of single hourglass network
        """
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.training = is_training

        self.input_image = tf.placeholder(tf.float32,shape=[None,256,256,3])
        # TODO how to preprocess data , make label  
        self.dataloader = dataloader
        
        #image width = 64,height = 64 , 16 joints = 16  
        self.out_tensor = tf.placeholder(tf.float32,shape=[None,64,64,7])
        self.output = self.generate_network(self.input_image,name = "single_model")
        
        self.sigmoid_out = tf.nn.sigmoid(self.output)
        
        self.loss = tf.reduce_mean(self.get_loss())
        
        self.opt = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate,decay=0.9).minimize(self.loss)
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        self.saver = tf.train.Saver()
    def generate_network(self,input_tensor,name="main_model"):
        """ generate main model 
        Args:
        input_tensor : input image 256x256x3 like in the paper
    	"""
        with tf.variable_scope(name):
            #256x256x3 -> 128x128x64
            conv1 = tf.layers.conv2d(input_tensor,filters=64,kernel_size=(7,7),strides=(2,2),padding="SAME")
            conv1 = relu_batch_norm(conv1)
            
            #128x128x64 -> 128x128x128
            pool1 = self.residual(conv1,128,"res1")
            

            #128x128x128 -> 64x64x128
            
            pool1 = maxpool(pool1)
            
            
            r1 = self.residual(pool1,128,name="r1")
            r2 = self.residual(r1,256,name="r2")

            hg = self.hourglass(r2,4,256,name="hourglass1")
            hg = self.hourglass(hg,4,256,name="hourglass2")
            drop = tf.layers.dropout(hg,rate = self.dropout_rate,training=self.training,name="dropout")
            
            ll =  tf.layers.conv2d(drop,filters=7,kernel_size=(1,1),strides=(1,1),padding="SAME")
            
            #return tf.nn.sigmoid(ll)
            return ll
    def hourglass(self,input_tensor,n,out_dim,name="hourglass"):
    	""" Hourglass block
    	Args:
    	    input_tensor :
    	    n            : number ofdown sampling step
    	    out_dim      : desired out dimension
    	    name         : to avoid tensorflow reuse error

    	"""
    	with tf.variable_scope(name):
            # upper 	
            up_1 = self.residual(input_tensor,out_dim,name="up_1")
            
            #low_ = tf.layers.MaxPooling2D(input_tensor,pool_size=2,strides=(1,1),padding="SAME")
            low_ = maxpool(input_tensor)
            low_1 = self.residual(low_,out_dim,name="low_1")
            
            if n > 0:
                #recursive loop
                low_2 = self.hourglass(low_1,n-1,out_dim,name="low"+str(n))
            else:
                low_2 = self.residual(low_1,out_dim,name="low_2")

            low_3 = self.residual(low_2,out_dim,name="low_3")
            
            up_2 = tf.image.resize_nearest_neighbor(low_3,tf.shape(low_3)[1:3]*2,name="upsampling")
            
            return tf.add(up_2,up_1) 

    def residual(self,input_tensor,out_dim,name="residual"):
        """ residual block same as Stacked hourglass network paper
	    Args:
		input_tensor :
		out_dim : desired filters out_dim
        """
        with tf.variable_scope(name):
     	    conv_block = self.conv_block(input_tensor,out_dim,name="conv_block")
     	    skip_layer = self.skip_layer(input_tensor,out_dim,name="skip_layer")
            
     	    out = tf.add(conv_block,skip_layer)
     	    return out
    # design is like torch code
    def conv_block(self,input_tensor,out_dim,name="conv_block"):
        """convolutional block 
        Args:
        input_tensor: input tensor (conv outputs )
        out_dim:  block feature 
        name :    scope name it has to different each time to avoid tensorflow error
        """
        with tf.variable_scope(name):
            norm_1 = relu_batch_norm(input_tensor)
            conv_1 = tf.layers.conv2d(norm_1,filters=out_dim/2,kernel_size=(1,1),strides=(1,1),padding="SAME")

            norm_2 = relu_batch_norm(conv_1)
            conv_2 = tf.layers.conv2d(conv_1,filters=out_dim/2,kernel_size=(3,3),strides=(1,1),padding="SAME")
            norm_3 = relu_batch_norm(conv_2)
            conv_3 = tf.layers.conv2d(norm_3,filters=out_dim,kernel_size=(1,1),strides=(1,1),padding="SAME")

            return conv_3


    #design is like torch code
    def skip_layer(self,input_tensor,out_dim,name="skip_layer"):
        """ Skip layer   
        Args:
            input_tensor:
            out_dim  :   Desired out_dim 
	    name :  to avoid tensorflow reuse error
        """
        with tf.variable_scope(name):
    	    if input_tensor.get_shape().as_list()[3] == out_dim:
                out = input_tensor
    	    else:
                out =  tf.layers.conv2d(input_tensor,filters=out_dim,kernel_size=(1,1),strides=(1,1),padding="SAME")

        return out
                    
    def get_loss(self):
    	#TODO  ->
        out = tf.contrib.layers.flatten(self.output)
        label = tf.contrib.layers.flatten(self.out_tensor)
    	return tf.nn.sigmoid_cross_entropy_with_logits(logits = out,labels=label)

    def train(self,epochs):
        #TODO
        #train method
        self.sess.run(tf.global_variables_initializer())
        for t in range(epochs):

            generator = self.dataloader()
            for (image,heatmap) in generator:
                loss,_ = self.sess.run([self.loss,self.opt],feed_dict={self.input_image:image,self.out_tensor:heatmap})

            if t % 10 == 0:
                print (loss)
                self.saver.save(self.sess,"./models/"+str(t)+".ckpt")


    def restore(self,model_path):
        self.saver.restore(self.sess,model_path)
### helper functions ####
def relu_batch_norm(x):
    return tf.nn.relu(tf.layers.batch_normalization(x))



if __name__=="__main__":

    print ("MAIN PROCESS")
    dataload = youtube(folder="./GT_frames",mat_file="YouTube_Pose_dataset.mat",batch_size=10) 
    
    model_ = model(dropout_rate=0.1,learning_rate=0.0001,dataloader=dataload)

    model_.train(10000)

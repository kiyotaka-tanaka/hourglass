import numpy as np 
import cv2
import os


class datagenerator:

        def __init__(self,image_folder,annotation,batch_size):
                #TODO ->
                self.folder = image_folder
                self.annotation_file = annotation
                self.batch_size = batch_size

                
	def __call__(self):
                #TODO -> yield input image , joint locations  
		#256x256x3  ,  64x64x16
		pass


        def make_gaussian_2D(self,size,center_x,center_y):
                #TODO -> make gaussian standard deviation 
                #np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

                


# helper function    
def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)



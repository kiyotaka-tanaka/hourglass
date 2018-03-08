mport numpy as np 
import cv2
import os

import scipy.io

"""
first version only for Youtube_pose_database
"""
class youtube:

    def __init__(self,folder,mat_file,batch_size):
        #TODO ->
        self.folder = folder
        self.batch_size = batch_size

        self.read_annotation(mat_file)


        
    def __call__(self):
        #TODO -> yield input image , joint locations  
	#256x256x3  ,  64x64x16
        
        for i,dat in enumerate(self.data[0]):
            video_name = dat["videoname"]
            file_path = os.path.join(self.folder,video_name[0])
            
            # get annotated frames
            image_ids = dat["frameids"][0]
            #sample batch size data
            samples = np.random.choice(np.arange(len(image_ids)),size=self.batch_size,replace=False)
            
            locations = dat["locs"]

            batch_image = []
            batch_heatmap = []
            
            for sample in samples:
                image_name ="frame_"+ str(image_ids[sample]).zfill(6)+".jpg"
                image_path = os.path.join(file_path,image_name)
               
                location = locations[:,:,sample] # 2x7 -> x,y   head , shoulder etc ...

                img_height,img_width = get_image_size(image_path)
                heat_map = []
                for i in range(7):
                    center = (int(location[0][i]*64.0/img_height),int(location[1][i]*64.0/img_width))

                    mapx = makeGaussian(64,center=center)
                    mapx = np.transpose(mapx)
                    heat_map.append(mapx)
                    
                batch_image.append(read_image(image_path))
                batch_heatmap.append(np.array(heat_map))
            batch_heatmap = np.array(batch_heatmap)
            batch_heatmap = np.swapaxes(batch_heatmap,1,3)
                
            yield batch_image,batch_heatmap
                
    def make_heatmap(self):
        """make heatmap from annotation 
        do not use for now 
        Args:
        """
        
        
        
        pass

    def get_locations(self):
        """
        do not use for now
        """
        
        pass

    
    def read_annotation(self,mat_path):
        self.data = read_mat(mat_path)
        
        

# helper functions    
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

def read_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img/255.0
    return cv2.resize(img,(256,256))
    


def read_mat(file_path):
    mat = scipy.io.loadmat(file_path)
    return mat["data"]


def get_image_size(img_path):
    img = cv2.imread(img_path)
    height , width =  img.shape[:2]

    return height, width

'''
This is a class for Facial Landmarks Detection model. 
'''

import numpy as np
from model import Model

class FacialLandmarksDetection(Model):
    '''
    Class for the Facial Landmarks Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        Model.__init__(self, model_name, device, extensions)

    def preprocess_output(self, outputs, threshold=None):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords = np.squeeze(outputs)
        for i in range(len(coords)):
            if i % 2 ==0 :
                coords[i] = int(coords[i]*self.img_w)
            else:
                coords[i] = int(coords[i]*self.img_h)
        return coords

    def getEyesCrop(self, coords):
        '''
        Return Left and right eye cropped images and their coordinates.
        '''
        left_eye_coord = [ abs(int(coords[0]-30)) , abs(int(coords[1]-30)), 
            abs(int(coords[0]+30)) , abs(int(coords[1]+30)) ]
        right_eye_coord =  [ abs(int(coords[2]-30)) , abs(int(coords[3]-30)),
            abs(int(coords[2]+30)) , abs(int(coords[3]+30)) ]

        left_eye_cropped = self.img[ left_eye_coord[1]:left_eye_coord[3],
            left_eye_coord[0]:left_eye_coord[2] ]
        right_eye_cropped = self.img[ right_eye_coord[1]:right_eye_coord[3], 
            right_eye_coord[0]:right_eye_coord[2] ]

        return left_eye_coord, right_eye_coord, left_eye_cropped, right_eye_cropped

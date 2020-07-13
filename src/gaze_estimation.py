'''
This is a class for Gaze Estimation model.
'''

import numpy as np
import cv2
from model import Model

class GazeEstimation(Model):
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        Model.__init__(self, model_name, device, extensions)
        
    def preprocess_input(self, gase_inputs):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        
        
        try:
            
            for eye in "left_eye_image","right_eye_image":

                p_frame = cv2.resize(gase_inputs[eye], \
                    (self.input_shape[eye][3], self.input_shape[eye][2]))
                p_frame = p_frame.transpose((2,0,1))
                p_frame = p_frame.reshape(1, *p_frame.shape)
                gase_inputs[eye] = p_frame
            
        except Exception as e:
            
            raise ValueError("Could preprocess the input frame...!",e)
            
        return gase_inputs

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        
        coord = np.squeeze(outputs)
        return coord
    
    
    def GetInputName(self):

        return self.network.inputs.keys()

    def GetInputShape(self):

        return {x:self.network.inputs[x].shape for x in self.input_name}
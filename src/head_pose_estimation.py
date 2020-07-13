'''
This is a class for Head Pose Estimation model.
'''

import numpy as np
from model import Model

class HeadPoseEstimation(Model):
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        Model.__init__(self, model_name, device, extensions)
        
    def predict(self, input_data):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_dict=input_data
    
        try:
            
            infer_request_handle = self.net_plugin.start_async(request_id=0, inputs=input_dict)
            status = infer_request_handle.wait(-1)
            
            if status == 0:
                
                outputs = {x:infer_request_handle.outputs[x]  for x in self.output_name}
                
        except Exception as e:
            
            raise ValueError("Could not perform inference...!",e)

        return outputs

    def preprocess_output(self, outputs, threshold=None):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords = []

        for key in  "angle_y_fc", "angle_p_fc","angle_r_fc":
            coords.append(np.squeeze(outputs[key]))
        
        return coords
    

    def GetOutputName(self):

        return self.network.outputs

    def GetOutputShape(self):

        return {x:self.network.outputs[x].shape for x in self.output_name}
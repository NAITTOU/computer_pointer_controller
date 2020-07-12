'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

class FacialLandmarksDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        
    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''

        model_xml = self.model_name+ ".xml"
        model_bin = self.model_name+".bin"

        try:
            self.network = IENetwork(model=model_xml, weights=model_bin)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        
        self.input_name=next(iter(self.network.inputs))
        self.input_shape=self.network.inputs[self.input_name].shape
        self.output_name=next(iter(self.network.outputs))
        self.output_shape=self.network.outputs[self.output_name].shape
 
        self.plugin = IECore()

        if not self.check_model():
            log.info("exiting program ...")
            exit(1)
        else:
            try:
                self.net_plugin = self.plugin.load_network(self.network, self.device, num_requests=1)
            except Exception as e:
                raise ValueError("Could not load the model...!",e)
            

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_dict={self.input_name:image}
    
        try:
            
            infer_request_handle = self.net_plugin.start_async(request_id=0, inputs=input_dict)
            status = infer_request_handle.wait(-1)
            
            if status == 0:
                
                outputs = infer_request_handle.outputs[self.output_name]   
                
        except Exception as e:
            
            raise ValueError("Could not perform inference...!",e)

        return outputs

    def check_model(self):

        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        
        ### TODO: Add any necessary extensions ###
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        if len(unsupported_layers) != 0:
            
            log.warning("Unsupported layers found: {}".format(unsupported_layers))
            
            if cpu_extension and "CPU" in self.device:
                
                log.info("Adding a CPU extension ...")
                self.plugin.add_extension(cpu_extension, self.device)
                log.info("The CPU extension was added")
                
                supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                
                if len(unsupported_layers) != 0:
                    
                    log.warning("There Still Unsuported layers even after the extension was added {}".format(unsupported_layers))
                    return False
                
                log.info("All the layers of you model are supported now by the Inference engine ")
                     
            else:
                
                log.ERROR("Check whether extensions are available to add to IECore.")
                log.info("exiting program ...")
                exit(1)
                return False
        else:
            log.info("All the layers are already supported!")

        return True

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        self.img = image
        self.img_w = image.shape[1]
        self.img_h = image.shape[0]
        try:
            
            p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)
            
        except Exception as e:
            
            raise ValueError("Could preprocess the input frame...!",e)
            
        return p_frame

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

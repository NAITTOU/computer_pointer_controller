'''
This is a class for Face Detection Model model. 
'''

from model import Model

class FaceDetection(Model):
    '''
    Class for the Face Detection Model.
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
        prevconf = 0 
        for box in outputs[0][0]:
            conf = box[2]
            if prevconf > conf:
                continue
            else:
                prevconf = conf

            if conf >= threshold :

                xmin = int(box[3] * self.img_w)
                ymin = int(box[4] * self.img_h)
                xmax = int(box[5] * self.img_w)
                ymax = int(box[6] * self.img_h)
            
                coord = [xmin,ymin,xmax,ymax]
        
        return coord

    def getFaceCrop(self, coord):
        '''
        Return Face cropped image.
        '''
        
        xmin = coord[0] 
        ymin = coord[1]
        xmax = coord[2]
        ymax = coord[3]
        cropped_face = self.img[ymin:ymax, xmin:xmax]

        return cropped_face

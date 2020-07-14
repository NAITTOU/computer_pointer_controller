"""Computer Pointer Controller."""

import os
import sys
import time
import socket
import json
import cv2

import logging as log
import numpy as np
import line_profiler
profile=line_profiler.LineProfiler()
import atexit
atexit.register(profile.print_stats)
import pprint

from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController

log.getLogger().setLevel(log.INFO)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--fd_model", required=True, type=str,
                        help="Path to the Face Detection model.")
    parser.add_argument("-hp", "--hp_model", required=True, type=str,
                        help="Path to the Head Pose Estimation model.")
    parser.add_argument("-fl", "--fl_model", required=True, type=str,
                        help="Path to the Facial Landmarks Detection model.")
    parser.add_argument("-ge", "--ge_model", required=True, type=str,
                        help="Path to the Gaze Estimation model.")
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="The type of input. Can be 'video' for video \
                        file, 'image' for image file,or 'cam' to use webcam feed")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file ,Leave empty for cam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-pi", "--print_output", type=str, default="N",
                        help="Show output of intermediate models for visualization"
                        " if yes type 'Y' ,Leave empty instead ")
    parser.add_argument("-ps", "--print_stats", type=str, default="N",
                        help="print the time it takes for each layer for each used model"
                        " if yes type 'Y' ,Leave empty instead ")
    return parser

def DisplayOutputs(
    batch, cropped_face, left_eye_cropped,
    right_eye_cropped, headp_coord, faced_coord,
    left_eye_coord, right_eye_coord, gaze_coord,Print_flag):
    '''
    Visualize intermediate models output.
    '''
        
    if Print_flag.lower() == "y" :

        cropped_face = cv2.resize(cropped_face, (300, 300))
        left_eye_cropped = cv2.resize(left_eye_cropped, (150, 150))
        right_eye_cropped = cv2.resize(right_eye_cropped, (150, 150)) 
        face_xy = cropped_face.shape[0]
        eyes_xy = right_eye_cropped.shape[0]
        y_unit = 40
        x_unit = 10
            
        fdLabelPosY = y_unit
        fdLabelPosX = x_unit
        cv2.putText(batch,"Face detection model output : ", (fdLabelPosX, fdLabelPosY), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        faceStartpPosY = fdLabelPosY+y_unit
        faceStartpPosX = x_unit
        faceEndpPosY = faceStartpPosY+face_xy
        faceEndpPosX = faceStartpPosX+face_xy
        batch[faceStartpPosY:faceEndpPosY, faceStartpPosX:faceEndpPosX] \
            = cropped_face

        flLabelPosY = faceEndpPosY+y_unit
        flLabelPosX = x_unit
        cv2.putText(batch,"Facial Landmarks Detection model output : ", (flLabelPosX, flLabelPosY), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        LeftStartpPosY = flLabelPosY+y_unit
        LeftStartpPosX = x_unit
        LeftEndpPosY = LeftStartpPosY+eyes_xy
        LeftEndpPosX = LeftStartpPosX+eyes_xy
        batch[LeftStartpPosY:LeftEndpPosY, LeftStartpPosX:LeftEndpPosX] = left_eye_cropped

        RightStartpPosY = flLabelPosY+y_unit
        RightStartpPosX = LeftEndpPosX + x_unit
        RightEndpPosY = LeftStartpPosY+eyes_xy
        RightEndpPosX = RightStartpPosX+eyes_xy
        batch[RightStartpPosY:RightEndpPosY, RightStartpPosX:RightEndpPosX] = right_eye_cropped
        hpLabelPosY = RightEndpPosY+y_unit
        hpLabelPosX = x_unit
        cv2.putText(batch,"Head Pose Estimation model output : ", (hpLabelPosX, hpLabelPosY), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,215,255), 2)
        
        hpPosY = hpLabelPosY+y_unit
        hpPosX = x_unit
        hpCoord = "yaw : {0:.2f} , pitch : {0:.2f} , roll : {0:.2f}".\
            format(headp_coord[0],headp_coord[1],headp_coord[2])
        cv2.putText(batch,hpCoord, (hpPosX, hpPosY), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,215,255), 2)

        geLabelPosY = hpPosY+y_unit
        geLabelPosX = x_unit
        cv2.putText(batch,"Gaze Estimation model output : ", (geLabelPosX, geLabelPosY), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (130,0,75), 2)
        
        gePosY = geLabelPosY+y_unit
        gePosX = x_unit
        geCoord = "x : {0:.2f} , y : {0:.2f} , z : {0:.2f}".\
            format(gaze_coord[0],gaze_coord[1],gaze_coord[2])
        cv2.putText(batch,geCoord, (gePosX, gePosY), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (130,0,75), 2)
        
        cv2.rectangle(batch, (faced_coord[0] , faced_coord[1]), \
            (faced_coord[2], faced_coord[3]),(0,0,255), 2)
        
        
    batch = cv2.resize(batch, (900,500))
    cv2.imshow("Gaze Computer Pointer Controller", batch)

def DisplayLayerwisePerformance(
        face_detection_model,Head_PoseEstimation_model,
        LandmarksDetection_model,GazeEstimation_model,print_stats
        ):
        '''
        Print the time it takes for each layer for each used model.
        '''
        
        if print_stats.lower() == "y" :
            pp = pprint.PrettyPrinter(indent=4)
            
            log.info("---------------------------------------------------")
            log.info("The time it takes for each layer in the" \
                "face-detection-adas-binary-0001 model")
            pp.pprint(face_detection_model.LayerwisePerformanceStats())
            log.info("---------------------------------------------------")

            log.info("---------------------------------------------------")
            log.info("The time it takes for each layer in the" \
                "head-pose-estimation-adas-0001 model")
            pp.pprint(Head_PoseEstimation_model.LayerwisePerformanceStats())
            log.info("---------------------------------------------------")

            log.info("---------------------------------------------------")
            log.info("The time it takes for each layer in the" \
                "landmarks-regression-retail-0009 model")
            pp.pprint(LandmarksDetection_model.LayerwisePerformanceStats())
            log.info("---------------------------------------------------")

            log.info("---------------------------------------------------")
            log.info("The time it takes for each layer in the" \
                "gaze-estimation-adas-0002 model")
            pp.pprint(GazeEstimation_model.LayerwisePerformanceStats())
            log.info("---------------------------------------------------")
        

@profile
def infer_on_stream(args):

    input_type = args.input_type
    input_file = args.input
    threshold = args.prob_threshold
    Print_flag = args.print_output
    print_stats = args.print_stats
    cpu_extension = args.cpu_extension
    device = args.device

    # Loading Face Detection model
    face_detection_model = FaceDetection(args.fd_model,device,cpu_extension)
    face_detection_model.load_model()

    # Loading Head Pose Estimation model
    Head_PoseEstimation_model = HeadPoseEstimation(args.hp_model,device,cpu_extension)
    Head_PoseEstimation_model.load_model()

    # Loading Facial Landmarks Detection model
    LandmarksDetection_model = FacialLandmarksDetection(args.fl_model,device,cpu_extension)
    LandmarksDetection_model.load_model()

    # Loading Gaze Estimation
    GazeEstimation_model = GazeEstimation(args.ge_model,device,cpu_extension)
    GazeEstimation_model.load_model()

    # Instantiating a pyautogui library object to control the mouse pointer
    MouseControll = MouseController("medium", "fast")

    feed=InputFeeder(input_type, input_file)
    feed.load_data()
    initial_h, initial_w, video_len, fps = feed.getCapInfo()
    
    # Start straming input
    for batch in feed.next_batch():

        # Running inference on Face Detection model to get the face coordinates
        faced_pframe = face_detection_model.preprocess_input(batch)
        faced_inputs={

          face_detection_model.input_name : faced_pframe,
          
        }
        faced_outputs = face_detection_model.predict(faced_inputs)
        faced_coord = face_detection_model.preprocess_output(faced_outputs,threshold)
        cropped_face = face_detection_model.getFaceCrop(faced_coord)

        # Running inference on Head Pose Estimation model to get head pose angle
        headp_pframe = Head_PoseEstimation_model.preprocess_input(cropped_face)
        headp_inputs={

          Head_PoseEstimation_model.input_name : headp_pframe,
          
        }
        headp_outputs = Head_PoseEstimation_model.predict(headp_inputs)
        headp_coord = Head_PoseEstimation_model.preprocess_output(headp_outputs)

        # Running inference on Facial Landmarks Detection model
        #  to get cropped left and right eye
        facial_pframe = LandmarksDetection_model.preprocess_input(cropped_face)
        facial_inputs={

          LandmarksDetection_model.input_name : facial_pframe,
          
        }
        facial_outputs = LandmarksDetection_model.predict(facial_inputs)
        facial_coords = LandmarksDetection_model.preprocess_output(facial_outputs)
        left_eye_coord, right_eye_coord, left_eye_cropped, right_eye_cropped = \
            LandmarksDetection_model.getEyesCrop(facial_coords)
        # Running inference Gaze Estimation model
        #  to get the coordinates of gaze direction vector
        gase_inputs={

          "left_eye_image" : left_eye_cropped,
          "right_eye_image" : right_eye_cropped,
          "head_pose_angles" : headp_coord

        }
        gase_pframes = GazeEstimation_model.preprocess_input(gase_inputs)
        gase_outputs = GazeEstimation_model.predict(gase_pframes)
        gaze_coord = GazeEstimation_model.preprocess_output(gase_outputs)
        
        DisplayOutputs(
            batch, cropped_face, left_eye_cropped,
            right_eye_cropped, headp_coord, faced_coord,
            left_eye_coord, right_eye_coord, gaze_coord,Print_flag)
        
        x = gaze_coord[0]
        y = gaze_coord[1]
        # Moving the mouse pointer according to x,y
        MouseControll.move(x, y)

    feed.close()

    DisplayLayerwisePerformance(
        face_detection_model,Head_PoseEstimation_model,
        LandmarksDetection_model,GazeEstimation_model,print_stats
        )

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    infer_on_stream(args)
    

if __name__ == '__main__':
    main()
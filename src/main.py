"""Computer Pointer Controller."""

import os
import sys
import time
import socket
import json
import cv2

import logging as log
import numpy as np

from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation

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
                        help="The type of input. Can be 'video' for video .\
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
    return parser

def infer_on_stream(args):

    input_type = args.input_type
    input_file = args.input
    threshold = args.prob_threshold

    # Loading Face Detection model
    face_detection_model = FaceDetection(args.fd_model)
    face_detection_model.load_model()

    # Loading Head Pose Estimation model
    Head_PoseEstimation_model = HeadPoseEstimation(args.hp_model)
    Head_PoseEstimation_model.load_model()

    # Loading Facial Landmarks Detection model
    LandmarksDetection_model = FacialLandmarksDetection(args.fl_model)
    LandmarksDetection_model.load_model()

    # Loading Gaze Estimation
    GazeEstimation_model = GazeEstimation(args.ge_model)
    GazeEstimation_model.load_model()

    feed=InputFeeder(input_type, input_file)
    feed.load_data()
    initial_h, initial_w, video_len, fps = feed.getCapInfo()
    
    # Start straming input
    for batch in feed.next_batch():

        # Running inference on Face Detection model to get the face coordinates
        faced_pframe = face_detection_model.preprocess_input(batch)
        faced_outputs = face_detection_model.predict(faced_pframe)
        faced_coord = face_detection_model.preprocess_output(faced_outputs,threshold)
        cropped_face = face_detection_model.getFaceCrop(faced_coord)

        # Running inference on Head Pose Estimation model to get head pose angle
        headp_pframe = Head_PoseEstimation_model.preprocess_input(cropped_face)
        headp_outputs = Head_PoseEstimation_model.predict(headp_pframe)
        headp_coord = Head_PoseEstimation_model.preprocess_output(headp_outputs)

        # Running inference on Facial Landmarks Detection model
        #  to get cropped left and right eye
        facial_pframe = LandmarksDetection_model.preprocess_input(cropped_face)
        facial_outputs = LandmarksDetection_model.predict(facial_pframe)
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

 

        cv2.rectangle(batch, (faced_coord[0] , faced_coord[1]), \
            (faced_coord[2], faced_coord[3]),(0,0,255), 1)

        cv2.rectangle(cropped_face, ( left_eye_coord[0], left_eye_coord[1] ), \
            ( left_eye_coord[2], left_eye_coord[3]), (0,255,0), 1)

        cv2.rectangle(cropped_face, (right_eye_coord[0], right_eye_coord[1]), \
            (right_eye_coord[2], right_eye_coord[3]), (0,255,0), 1)
        
        cv2.imshow('Frame',batch)



    out_video.release()
    feed.close()

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
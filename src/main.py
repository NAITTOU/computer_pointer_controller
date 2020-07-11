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
                        help="Path to Head Pose Estimation model.")
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

    face_detection_model = FaceDetection(args.fd_model)
    face_detection_model.load_model()

    Head_PoseEstimation_model = HeadPoseEstimation(args.hp_model)
    Head_PoseEstimation_model.load_model()
    
    feed=InputFeeder(input_type, input_file)
    feed.load_data()
    initial_h, initial_w, video_len, fps = feed.getCapInfo()
    output_path = "/home/naittoulahyane/projects/intel_edge_ai_iot_developers/computer_pointer_controller/output_video.mp4"
    print(output_path)
    #out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'),\
                #cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    out_video = cv2.VideoWriter(output_path, 0x00000021, fps, (300, 400))
    
    for batch in feed.next_batch():

        fd_pframe = face_detection_model.preprocess_input(batch)
        fd_outputs = face_detection_model.predict(fd_pframe)
        fd_coord = face_detection_model.preprocess_output(fd_outputs,threshold)
        print(fd_coord)
        xmin = int(fd_coord[0] * initial_w)
        ymin = int(fd_coord[1] * initial_h)
        xmax = int(fd_coord[2] * initial_w)
        ymax = int(fd_coord[3] * initial_h)
        fd_batch = batch[ymin:ymax, xmin:xmax]

        hp_pframe = Head_PoseEstimation_model.preprocess_input(fd_batch)
        hp_outputs = Head_PoseEstimation_model.predict(hp_pframe)
        hp_coords = Head_PoseEstimation_model.preprocess_output(hp_outputs)
 
        out_video.write(cv2.resize(fd_batch, (300, 400)))

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
import argparse
import os, sys
from rekognition.pipeline.pipeline import Pipeline

# Import pipeline elements
# Data Handlers
from rekognition.pipeline.input_handlers.video_handler import VideoHandlerElem

# Computer Vision
from rekognition.pipeline.face_detectors.mobilenets_ssd import MobileNetsSSDFaceDetector
from rekognition.pipeline.recognizers.facenet_recognizer import FacenetRecognizer

# Output
from rekognition.pipeline.output_handlers.json_handler import JSONHandler
from rekognition.pipeline.output_handlers.videooutput_handler import VideoOutputHandler

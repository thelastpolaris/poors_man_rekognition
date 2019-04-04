import argparse
import os, sys
from rekognition.pipeline.pipeline import Pipeline

# Import pipeline elements
# Data Handlers
from rekognition.pipeline.video_handler import VideoHandlerElem
from rekognition.pipeline.image_handler import ImageHandlerElem

# Computer Vision
from rekognition.pipeline.mobilenets_ssd import MobileNetsSSDFaceDetector
from rekognition.pipeline.yolov3_face_detector import YOLOv3FaceDetector
from rekognition.pipeline.facenet_recognizer import FacenetRecognizer

# Output
from rekognition.pipeline.json_handler import JSONHandler
from rekognition.pipeline.videooutput_handler import VideoOutputHandler
from rekognition.pipeline.imageoutput_handler import ImageOutputHandler

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="Either video or image")
args = vars(ap.parse_args())

input_path = fileDir + "/" + args["input"]

if os.path.isfile(input_path) != True and os.path.isdir(input_path) != True :
	print("Input file/dir doesn't exist. Terminating.")
	sys.exit()

# create pipeline
p = Pipeline()

image = False
# Data handlers
if image:
	datahandler = ImageHandlerElem()
else:
	datahandler = VideoHandlerElem()

# Face Detector
face_detector = MobileNetsSSDFaceDetector(min_score_thresh=.5)
# face_detector = YOLOv3FaceDetector(min_score_thresh=.3)

# Face Recognizer
face_recognizer = FacenetRecognizer(fileDir + "/rekognition/model/facenet_20180408.pb", fileDir + "/rekognition/model/facenet_classifier.pkl")

# Output Handler
jsonhandler = JSONHandler()
if image:
	output_hand = ImageOutputHandler()
else:
	output_hand = VideoOutputHandler()

# Construct the pipeline
p.add_element(datahandler, input_path)
p.add_element(face_detector, datahandler)
p.add_element(face_recognizer, face_detector)
p.add_element(jsonhandler, face_recognizer)
p.add_element(output_hand, jsonhandler)

# Print the pipeline
print(p)

# Run the pipeline
p.run()

# frames_rgb = HandleVideoElem.extract_keyframes(input_path)
# print(len(frames_rgb))
# print(frames_rgb[0].shape)
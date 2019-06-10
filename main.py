import argparse
import os, sys
from rekognition.pipeline.pipeline import Pipeline
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Import pipeline elements
# Data Handlers
from rekognition.pipeline.input_handlers.video_handler import VideoHandlerElem
from rekognition.pipeline.input_handlers.image_handler import ImageHandlerElem

# Computer Vision
from rekognition.pipeline.face_detectors.mobilenets_ssd import MobileNetsSSDFaceDetector
from rekognition.pipeline.face_detectors.yolov3_face_detector import YOLOv3FaceDetector
from rekognition.pipeline.face_detectors.mtcnn import MTCNNFaceDetector
from rekognition.pipeline.recognizers.facenet_recognizer import FacenetRecognizer

# Output
from rekognition.pipeline.output_handlers.json_handler import JSONHandler
from rekognition.pipeline.output_handlers.videooutput_handler import VideoOutputHandler
from rekognition.pipeline.output_handlers.imageoutput_handler import ImageOutputHandler

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

image = False
# Data handlers
if image:
	datahandler = ImageHandlerElem(input_path)
else:
	datahandler = VideoHandlerElem(input_path)

datahandler.max_frames = 50

# Face Detector
face_detector = MobileNetsSSDFaceDetector()
# face_detector = YOLOv3FaceDetector()
# face_detector = MTCNNFaceDetector()

# Face Recognizer
face_recognizer = FacenetRecognizer(fileDir + "/rekognition/model/facenet_20180408.pb", fileDir + "/rekognition/model/pozner.pkl")

# Output Handler
jsonhandler = JSONHandler("test")
if image:
	output_hand = ImageOutputHandler("test")
else:
	output_hand = VideoOutputHandler("test")

# create pipeline
p = Pipeline([datahandler,
			  face_detector,
			  face_recognizer,
			  output_hand])

# Print the pipeline
print(p)

# Run the pipeline
p.run()

# frames_rgb = HandleVideoElem.extract_keyframes(input_path)
# print(len(frames_rgb))
# print(frames_rgb[0].shape)
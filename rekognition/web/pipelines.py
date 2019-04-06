# Import pipeline elements
from ..pipeline.pipeline import Pipeline

# Data Handlers
from ..pipeline.input_handlers.video_handler import VideoHandlerElem
from ..pipeline.input_handlers.image_handler import ImageHandlerElem

# Computer Vision
from ..pipeline.face_detectors.mobilenets_ssd import MobileNetsSSDFaceDetector
from ..pipeline.face_detectors.yolov3_face_detector import YOLOv3FaceDetector
from ..pipeline.recognizers.facenet_recognizer import FacenetRecognizer

# Output
from ..pipeline.output_handlers.json_handler import JSONHandler
from ..pipeline.output_handlers.videooutput_handler import VideoOutputHandler
from ..pipeline.output_handlers.imageoutput_handler import ImageOutputHandler

import os

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

def createPipeline(input_path, isImage = False, useYolo = False):
	p = Pipeline()
	
	# Data handlers
	if isImage:
		datahandler = ImageHandlerElem()
	else:
		datahandler = VideoHandlerElem()
	
	# Face Detector
	if useYolo:
		face_detector = YOLOv3FaceDetector(min_score_thresh=.5)
	else:
		face_detector = MobileNetsSSDFaceDetector(min_score_thresh=.5)
	
	# Face Recognizer
	face_recognizer = FacenetRecognizer(parentDir + "/model/facenet_20180408.pb", parentDir + "/model/facenet_classifier.pkl")
	
	# Output Handler
	jsonhandler = JSONHandler()
	if isImage:
		output_hand = ImageOutputHandler()
	else:
		output_hand = VideoOutputHandler()
	
	# Construct the pipeline
	p.add_element(datahandler, input_path)
	p.add_element(face_detector, datahandler)
	p.add_element(face_recognizer, face_detector)
	p.add_element(jsonhandler, face_recognizer)
	p.add_element(output_hand, jsonhandler)
	return p
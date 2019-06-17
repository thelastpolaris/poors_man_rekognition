import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Import pipeline elements
from rekognition.pipeline.pipeline import Pipeline
# Data Handlers
from rekognition.pipeline.input_handlers.preprocessors import ResizeImage, InvertColors, Lambda
from rekognition.pipeline.input_handlers.video_handler import VideoHandlerElem

import tensorflow

# Computer Vision
from rekognition.pipeline.face_detectors.face_detector import FaceDetectorElem
from rekognition.pipeline.face_detectors.mobilenets_ssd import MobileNetsSSDFaceDetector
from rekognition.pipeline.face_detectors.yolov3_face_detector import YOLOv3FaceDetector
from rekognition.pipeline.face_detectors.mtcnn import MTCNNFaceDetector

from rekognition.pipeline.face_detectors.dsfd import DSFDFaceDetector

from rekognition.pipeline.recognizers.face_recognizer import FaceRecognizerElem
from rekognition.pipeline.recognizers.facenet_recognizer import FacenetRecognizer

# Output
from rekognition.pipeline.output_handlers.videooutput_handler import VideoOutputHandler

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__)) + "/"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Either video or image")

args = vars(ap.parse_args())

input_path = args["input"]
filename = os.path.basename(input_path)
filename_wo_ext = os.path.splitext(filename)[0]

resizer = ResizeImage(320, 240)
invert = InvertColors()
lambd = Lambda(lambda image: image)

datahandler = VideoHandlerElem([resizer])

# face_detector = FaceDetectorElem(MobileNetsSSDFaceDetector())
# face_detector = FaceDetectorElem(YOLOv3FaceDetector())
# face_detector = FaceDetectorElem(DSFDFaceDetector())
face_detector = FaceDetectorElem(MTCNNFaceDetector())

face_recognizer = FaceRecognizerElem(FacenetRecognizer(fileDir + "/rekognition/model/facenet_20180408.pb", fileDir + "/rekognition/model/pozner.pkl"))
output_hand = VideoOutputHandler()

pipeline = Pipeline([datahandler,
                     face_detector,
                     face_recognizer,
                     output_hand
                     ])

print(pipeline)

# Benchmarks stuff
benchmark_boxes = fileDir + "test/videos/face_detection/benchmark_boxes/" + filename_wo_ext + '.xml'
# benchmark_boxes = ""

embeddings_path = "s"

pipeline.run({datahandler: {"input_path" : input_path, "max_frames" : 0},
              face_detector: {"min_score": 0.6, "benchmark": True, "benchmark_boxes": benchmark_boxes},
              output_hand: {"output_name": filename_wo_ext + "_" + face_detector.__str__()}})

datahandler = VideoHandlerElem([ResizeImage(320, 240)])
face_detector = FaceRecognizerElem(YOLOv3FaceDetector())
face_recognizer = FaceRecognizerElem(FacenetRecognizer())
output_hand = VideoOutputHandler()

pipeline = Pipeline([datahandler,
					 face_detector,
					 face_recognizer,
					 output_hand])

pipeline.run({datahandler: {"input_path" : input_path},
			  face_detector: {"min_score": 0.6},
			  face_recognizer: {"embeddings_path" : embeddings_path},
			  output_hand: {"output_name": filename_wo_ext }})
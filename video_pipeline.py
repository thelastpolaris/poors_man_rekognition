import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Import pipeline elements
from rekognition.pipeline.pipeline import Pipeline
# Data Handlers
from rekognition.pipeline.input_handlers.preprocessors import ResizeImage, InvertColors, Lambda
from rekognition.pipeline.input_handlers.video_handler import VideoHandlerElem

# Computer Vision
from rekognition.pipeline.face_detectors.face_detector import FaceDetectorElem
from rekognition.pipeline.face_detectors.mobilenets_ssd import MobileNetsSSDFaceDetector
from rekognition.pipeline.face_detectors.yolov3_face_detector import YOLOv3FaceDetector

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

resizer = ResizeImage(640, 480)
invert = InvertColors()
lambd = Lambda(lambda image: image)

datahandler = VideoHandlerElem(input_path, [resizer])
# datahandler.max_frames = 1000

face_detector = FaceDetectorElem(MobileNetsSSDFaceDetector(min_score=.5))
# face_detector = FaceDetectorElem(YOLOv3FaceDetector(min_score=.5))
face_recognizer = FaceRecognizerElem(FacenetRecognizer(fileDir + "/rekognition/model/facenet_20180408.pb", fileDir + "/rekognition/model/pozner.pkl"))
output_hand = VideoOutputHandler("test2")

pipeline = Pipeline([datahandler,
                     face_detector,
                     face_recognizer,
                     output_hand])

print(pipeline)

pipeline.run()
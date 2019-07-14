import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Import pipeline elements
from rekognition.pipeline.pipeline import Pipeline
# Data Handlers
from rekognition.pipeline.input_handlers.preprocessors import ResizeImage, InvertColors, Lambda
from rekognition.pipeline.input_handlers.video_handler import VideoHandlerElem

# Computer Vision
from rekognition.pipeline.similar_frames.similar_frames_finder import SimilarFramesFinder
from rekognition.pipeline.similar_frames.comp_hist_kernel import CompHist
from rekognition.pipeline.similar_frames.ssim_kernel import SSIM


from rekognition.pipeline.face_detectors.face_detector import FaceDetectorElem
from rekognition.pipeline.face_detectors.mobilenets_ssd import MobileNetsSSDFaceDetector
from rekognition.pipeline.face_detectors.yolov3_face_detector import YOLOv3FaceDetector
from rekognition.pipeline.face_detectors.mtcnn import MTCNNFaceDetector
from rekognition.pipeline.face_detectors.dsfd import DSFDFaceDetector

from rekognition.pipeline.recognizers.face_recognizer import FaceRecognizerElem
from rekognition.pipeline.recognizers.facenet_recognizer import FacenetRecognizer
from rekognition.pipeline.recognizers.arcface_recognizer import ArcFaceRecognizer

# Output
from rekognition.pipeline.output_handlers.videooutput_handler import VideoOutputHandler

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(absFilePath) + "/"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Either video or image")

args = vars(ap.parse_args())

input_path = args["input"]
filename = os.path.basename(input_path)
filename_wo_ext = os.path.splitext(filename)[0]

resizer = ResizeImage(640, 480)
invert = InvertColors()
lambd = Lambda(lambda image: image)

datahandler = VideoHandlerElem()

# Group similar frames
simframes = SimilarFramesFinder(CompHist())
# simframes = SimilarFramesFinder(SSIM())

# Face Detectors
# face_detector = FaceDetectorElem(MobileNetsSSDFaceDetector())
face_detector = FaceDetectorElem(YOLOv3FaceDetector())
# face_detector = FaceDetectorElem(DSFDFaceDetector())
# face_detector = FaceDetectorElem(MTCNNFaceDetector())

# face_recognizer = FaceRecognizerElem(ArcFaceRecognizer(fileDir + "/rekognition/model/arcface/classifiers/arcface_first_evals_scikit_aug.pkl"))
face_recognizer = FaceRecognizerElem(FacenetRecognizer(fileDir + "/rekognition/model/facenet/classifiers/facenet_first_evals_scikit_aug.pkl"))
output_hand = VideoOutputHandler()

pipeline = Pipeline([datahandler,
                     simframes,
                     face_detector,
                     face_recognizer,
                     output_hand
                     ])
print(pipeline)

# Benchmarks stuff
benchmark_boxes = fileDir + "test/videos/benchmark_boxes/" + filename_wo_ext + '.xml'
# benchmark_boxes = None
out_name = "{}_{}_{}".format(filename_wo_ext, face_detector, face_recognizer)

pipeline.run({datahandler: {"input_path" : input_path, "max_frames" : 100, "preprocessors": [resizer]},
              simframes: {"sim_threshold": 0.99, "max_jobs": 10},
              face_detector: {"min_score": 0.6, "benchmark_boxes": benchmark_boxes},
              face_recognizer: {"backend":"SciKit", "n_ngbr": 10, "benchmark_boxes": benchmark_boxes, "distance_threshold": 0.6},
              output_hand: {"output_name": out_name},
              pipeline: {"out_name": "output/" + out_name}}, benchmark=True)
import os, sys
import argparse

from rekognition.pipeline.recognizers.arcface_recognizer import ArcFaceRecognizer
from rekognition.pipeline.recognizers.facenet_recognizer import FacenetRecognizer

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="Face images folder")
args = vars(ap.parse_args())

input_path = args["input"]

# ArcFaceRecognizer().train(input_path, "arcface_first_evals_scikit.pkl", batch_size = 16, backend="SciKit")
FacenetRecognizer().train(input_path, "facenet_final_report_scikit.pkl", batch_size = 100, backend="SciKit")
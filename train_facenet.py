import os, sys
from rekognition.pipeline.pipeline import Pipeline
import argparse

from rekognition.pipeline.facenet_recognizer import FacenetRecognizer

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="Either video or image")
args = vars(ap.parse_args())

input_path = fileDir + "/" + args["input"]

FacenetRecognizer().train(input_path, "facenet_classifier.pkl")
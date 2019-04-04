import os, sys, cv2
import numpy as np
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

input_path = fileDir + "/" + args["input"] + "/"

input_data = []

img_filenames = os.listdir(input_path)

for filename in img_filenames:
	# image = cv2.imread(input_path + filename)
	input_data.append(str(input_path + filename))

FacenetRecognizer().run(input_data)
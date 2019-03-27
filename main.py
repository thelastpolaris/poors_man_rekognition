import argparse
import os, sys
from rekognition.pipeline.pipeline import Pipeline

# Import pipeline elements
from rekognition.pipeline.handle_video import HandleVideoElem
from rekognition.pipeline.mobilenets_ssd import MobileNetsSSDFaceDetector

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="Either video or image")
args = vars(ap.parse_args())

if os.path.isfile(args["input"]) != True:
	print("Input file doesn't exist. Terminating.")
	sys.exit()

# create pipeline
p = Pipeline()

# create first element to handle video
video = HandleVideoElem()
face_detector = MobileNetsSSDFaceDetector()

p.add_element(video, args["input"])
p.add_element(face_detector, video)

# Print the pipeline
print(p)

# Run the pipeline
p.run()

# frames_rgb = HandleVideoElem.extract_keyframes(args["input"])
# print(len(frames_rgb))
# print(frames_rgb[0].shape)
import argparse
import os, sys
from rekognition.pipeline.pipeline import Pipeline

# Import pipeline elements
# Data Handlers
from rekognition.pipeline.video_handler import VideoHandlerElem
from rekognition.pipeline.image_handler import ImageHandlerElem

# Computer Vision
from rekognition.pipeline.mobilenets_ssd import MobileNetsSSDFaceDetector

# Output

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
image = ImageHandlerElem()
# video = VideoHandlerElem()
face_detector = MobileNetsSSDFaceDetector()
# face_recognition = 

p.add_element(image, args["input"])
# p.add_element(video, args["input"])
p.add_element(face_detector, image)

# Print the pipeline
print(p)

# Run the pipeline
p.run()

# frames_rgb = HandleVideoElem.extract_keyframes(args["input"])
# print(len(frames_rgb))
# print(frames_rgb[0].shape)
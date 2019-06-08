import os
from .face_detector_kernel import FaceDetectorKernel
from mtcnn.mtcnn import MTCNN
import numpy as np

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class MTCNNFaceDetector(FaceDetectorKernel):
	def __init__(self):
		super().__init__()
		self._detector = None

	def load_model(self):
		self._detector = MTCNN()

	def inference(self, image):
		image = image[:, :, [2, 1, 0]]
		results = self._detector.detect_faces(image)

		scores = []
		boxes = []

		for res in results:
			scores.append(res["confidence"])

			# MTCNN returns box as as [x, y, width, height]
			# and we need to convert it to ymin, xmin, ymax, xmax representation
			box = np.array(res["box"])

			ymin = box[1]
			xmin = box[0]
			ymax = box[1] + box[3]
			xmax = box[0] + box[2]

			box = np.array([ymin, xmin, ymax, xmax])

			boxes.append(box)

		return scores, boxes
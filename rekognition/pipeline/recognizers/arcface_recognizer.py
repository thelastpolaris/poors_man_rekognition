# Part of the code used is given under MIT License
# Copyright (c) 2018 Jiankang Deng and Jia Guo

from .face_recognizer_kernel import FaceRecognizerKernel
import os, math, cv2
import numpy as np
from ...model.arcface.face_model import FaceModel
import facenet.src.facenet as facenet
from progress.bar import Bar

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class ArcFaceRecognizer(FaceRecognizerKernel):
	def __init__(self, arcface_classifier=parentDir):
		super().__init__()
		self._arcface_model = parentDir + "/../model/arcface/model-r100-ii/model,0"
		self._classifier = arcface_classifier
		self._embedding_size = 512
		self._model = None
		self._normalize_image = False
		self._preprocess_batch = True
		self._image_size = 112

	def load_model(self):
		self._model = FaceModel(self._arcface_model)

	def preprocess_face(self, face_img):
		return self._model.get_input(face_img)

	def calculate_embeddings(self, face_img):
		return np.float32(self._model.get_feature(np.float32(face_img)))
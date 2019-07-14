# Part of the code used is given under MIT License
# Copyright (c) 2019 https://github.com/yu4u/age-gender-estimation

from .face_age_gender_kernel import FaceAgeGenderKernel
import os, cv2
import numpy as np
from ...model.dex.wide_resnet import WideResNet

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class DEXAgeGenderKernel(FaceAgeGenderKernel):
	def __init__(self):
		super().__init__()
		self._dex_model = parentDir + "/../model/dex/weights.28-3.73.hdf5" # path to weight file (e.g. weights.28-3.73.hdf5)
		# Model parameters
		self._depth = 16 # depth of network
		self._width = 8 # width of network
		self._image_size = 64 # size to which every face will be resized
		self._margin = 0.4 # margin around detected face for age-gender estimation
		self._model = None

	def load_model(self):
		self._model = WideResNet(self._image_size, depth=self._depth, k=self._width)()
		self._model.load_weights(self._dex_model)

	def preprocess(self, faces_img):
		faces = np.empty((len(faces_img), self._image_size, self._image_size, 3))

		if len(faces_img) > 0:
			for i, d in enumerate(faces_img):
				processed_img = cv2.resize(faces_img[i], dsize=(self._image_size, self._image_size), interpolation=cv2.INTER_CUBIC)
				faces[i, :, :, :] = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

		return faces

	def get_age_gender(self, faces_img):
		results = self._model.predict(faces_img)

		# Gender
		genders = results[0]
		predicted_genders = []
		for gender in genders:
			predicted_genders.append("M" if gender[0] < 0.5 else "F")

		# Age
		ages = np.arange(0, 101).reshape(101, 1)
		predicted_ages = np.ceil(results[1].dot(ages).flatten()).astype(int).tolist()

		return predicted_ages, predicted_genders
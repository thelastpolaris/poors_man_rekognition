# Part of the code used is given under MIT License
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

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

	def load_model(self):
		self._model = FaceModel(self._arcface_model)

	def preprocess_face(self, img):
		return self._model.get_input(img)

	def calculate_embeddings(self, faces, data_from_pipeline=True, batch_size=100, image_size=160):
		emb_array = None
		i = 0

		if data_from_pipeline:
			emb_array = []
			i += 1

			if len(faces):
				for face in faces:
					img = self.preprocess_face(face)
					emb_array.append(self._model.get_feature(img))

			emb_array = np.float32(emb_array)
		else:
			nrof_images = len(faces)
			nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
			emb_array = np.zeros((nrof_images, self._embedding_size))
			bar = Bar('Processing', max= nrof_images)

			for i in range(nrof_batches_per_epoch):
				start_index = i * batch_size
				end_index = min((i + 1) * batch_size, nrof_images)
				paths_batch = faces[start_index:end_index]
				images = facenet.load_data(paths_batch, False, False, image_size, False)
				images = self.preprocess_face(images)
				emb_array[start_index:end_index, :] = self._model.get_feature(images)
				bar.next(end_index - start_index)

		return emb_array
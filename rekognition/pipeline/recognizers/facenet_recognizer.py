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
import tensorflow as tf
import facenet.src.facenet as facenet
import os, math, cv2
import numpy as np

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class FacenetRecognizer(FaceRecognizerKernel):
	def __init__(self, facenet_classifier=parentDir):
		super().__init__()
		self._facenet_model = parentDir + "/../model/facenet/facenet_20180408.pb"
		self._classifier = facenet_classifier
		self._image_size = 160

	def load_model(self):
		self._graph = tf.Graph()
		self._sess = tf.Session()

		# Load the model
		with self._sess.as_default():
			facenet.load_model(self._facenet_model)

		self._images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		self._embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		self._phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		self._embedding_size = self._embeddings.get_shape()[1]

	def preprocess_face(self, face_img):
		face_img = cv2.resize(face_img, dsize=(self._image_size, self._image_size), interpolation=cv2.INTER_CUBIC)

		face_img = facenet.prewhiten(face_img)
		face_img = facenet.crop(face_img, False, self._image_size)
		face_img = facenet.flip(face_img, False)

		return face_img

	def calculate_embeddings(self, face_img):
		feed_dict = {self._images_placeholder: np.float32(face_img), self._phase_train_placeholder: False}
		emb_array = self._sess.run(self._embeddings, feed_dict=feed_dict)

		return np.float32(emb_array)
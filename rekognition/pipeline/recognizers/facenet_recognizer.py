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
		self._facenet_model = parentDir + "/../model/facenet_20180408.pb"
		self._classifier = facenet_classifier

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

	def calculate_embeddings(self, faces, data_from_pipeline=True, batch_size=100, image_size=160):
		emb_array = None

		i = 0

		if data_from_pipeline:
			i += 1

			if len(faces):
				face_images = []

				for face in faces:
					img = face

					img = cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)

					img = facenet.prewhiten(img)
					img = facenet.crop(img, False, image_size)
					img = facenet.flip(img, False)

					face_images.append(img)

				feed_dict = {self._images_placeholder: np.array(face_images), self._phase_train_placeholder: False}
				emb_array = self._sess.run(self._embeddings, feed_dict=feed_dict)
		else:
			nrof_images = len(faces)
			nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
			emb_array = np.zeros((nrof_images, self._embedding_size))

			for i in range(nrof_batches_per_epoch):
				start_index = i * batch_size
				end_index = min((i + 1) * batch_size, nrof_images)
				paths_batch = faces[start_index:end_index]
				images = facenet.load_data(paths_batch, False, False, image_size)
				feed_dict = {self._images_placeholder: images, self._phase_train_placeholder: False}
				emb_array[start_index:end_index, :] = self._sess.run(self._embeddings, feed_dict=feed_dict)


		return emb_array
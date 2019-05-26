# Part of the code used is given under MIT License
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

from progress.bar import Bar
from .face_recognizer import FaceRecognizerElem
import tensorflow as tf
import facenet.src.facenet as facenet
import pickle
from sklearn.neighbors import NearestNeighbors
import os, math, cv2
import numpy as np
from numba import cuda
from scipy.misc import imresize
from collections import Counter

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class FacenetRecognizer(FaceRecognizerElem):
	def __init__(self, facenet_model = parentDir + "/model/facenet_20180408.pb", facenet_classifier = parentDir + "/model/facenet_classifier.pkl"):
		super().__init__
		self._facenet_model = facenet_model
		self._facenet_classifier = facenet_classifier

		self._graph = tf.Graph()
		self._sess = tf.Session()

		# Load the model
		with self._sess.as_default():
			facenet.load_model(self._facenet_model)

	def calculate_embeddings(self, input_data, data_from_pipeline = True, batch_size = 100, image_size = 160):
		images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		embedding_size = embeddings.get_shape()[1]

		emb_array = None

		# print('Calculating features for images')

		# bar = Bar('Processing', max = len(input_data))
		i = 0

		if data_from_pipeline:
			data = input_data
			i += 1
			faces = data.faces

			if len(faces):
				face_images = []

				emb_array = np.zeros((len(faces), embedding_size))

				for face in faces:
					img = face.face_image

					img = cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)

					img = facenet.prewhiten(img)
					img = facenet.crop(img, False, image_size)
					img = facenet.flip(img, False)

					face_images.append(img)

				feed_dict = { images_placeholder: np.array(face_images), phase_train_placeholder: False}
				emb_array = self._sess.run(embeddings, feed_dict=feed_dict)

			# bar.next()
		else:
			nrof_images = len(input_data)
			nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
			emb_array = np.zeros((nrof_images, embedding_size))

			for i in range(nrof_batches_per_epoch):
				start_index = i*batch_size
				end_index = min((i+1)*batch_size, nrof_images)
				paths_batch = input_data[start_index:end_index]
				images = facenet.load_data(paths_batch, False, False, image_size)
				feed_dict = { images_placeholder:images, phase_train_placeholder:False }
				emb_array[start_index:end_index,:] = self._sess.run(embeddings, feed_dict=feed_dict)

				# bar.goto(bar.index + end_index - start_index)

		# bar.finish()

		return emb_array

	def train(self, dataset_folder, model_name):
		dataset = facenet.get_dataset(dataset_folder)

		# Check that there are at least one training image per class
		# for cls in dataset:
			# assert(len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')            

		paths, labels = facenet.get_image_paths_and_labels(dataset)

		print('Number of classes: %d' % len(dataset))
		print('Number of images: %d' % len(paths))

		print("Calculating embeddings for new data")
		data_emb = self.calculate_embeddings(paths, False)

		class_names = [ cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
		with open(model_name, 'wb') as outfile:
			pickle.dump((data_emb, class_names, labels), outfile)
		print('Saved classifier model to file "%s"' % model_name)

	def run(self, input_data):
		print("Recognizing the face")

		infile = open(self._facenet_classifier, 'rb')
		(model_emb, class_names, labels) = pickle.load(infile)
		print('Loaded classifier model from file "%s"' % self._facenet_classifier)
		
		n_ngbr = 10
		nbrs = NearestNeighbors(n_neighbors=n_ngbr, algorithm='ball_tree').fit(model_emb)
		
		# bar = Bar('Processing', max = len(input_data))

		for data in input_data:
			emb_array = self.calculate_embeddings(data)

			faces = data.faces

			if len(faces):
				distances, indices = nbrs.kneighbors(emb_array)
				
				for f in range(len(faces)):
					inds = indices[f]
					classes = np.array([labels[i] for i in inds])
					label = Counter(classes).most_common(1)[0][0]

					person_name = class_names[label]
					confidence = np.sum(classes == label)/n_ngbr
					
					if confidence <= 0.3:
						person_name = "Unknown"

					faces[f].set_person(person_name, confidence)

				yield data

			# bar.next()

		# bar.finish()

		# return input_data
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
from sklearn.svm import SVC
import os, math, cv2
import numpy as np

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class FacenetRecognizer(FaceRecognizerElem):
	__facenet_model = parentDir + "/model/facenet_20180408.pb"
	__facenet_classifier = parentDir + "/model/facenet_classifier.pkl"

	def __init__(self):
		super().__init__

	def train(self):
		pass

	def calculate_embeddings(self, input_data):
		tf.reset_default_graph()
		with tf.Graph().as_default():
			sess = tf.Session()
			# Load the model
			facenet.load_model(self.__facenet_model)

			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			embedding_size = embeddings.get_shape()[1]

			data_emb = []
			
			print('Calculating features for images')

			bar = Bar('Processing', max = len(input_data))

			for data in input_data:
				faces = data.get_faces()
				emb_array = None
				
				if len(faces):
					face_images = []

					emb_array = np.zeros((len(faces), embedding_size))

					for face in faces:
						face_images.append(cv2.resize(face.get_face_image(), dsize=(160, 160), interpolation=cv2.INTER_CUBIC))		

					feed_dict = { images_placeholder: np.array(face_images), phase_train_placeholder: False}
					emb_array = sess.run(embeddings, feed_dict=feed_dict)

				data_emb.append(emb_array)
									
				bar.next()

			bar.finish()

		return data_emb

	def run(self, input_data):
		print("Recognizing the face")
		emb_array = self.calculate_embeddings(input_data)

		infile = open(self.__facenet_classifier, 'rb')
		(model, class_names) = pickle.load(infile)

		for d in range(len(input_data)):
			faces = input_data[d].get_faces()

			if len(faces):
				# print('Loaded classifier model from file "%s"' % classifier_filename_exp)
				predictions = model.predict_proba(emb_array[d])
					
				best_class_indices = np.argmax(predictions, axis=1)
				best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

				for f in range(len(faces)):
					faces[f].set_person(class_names[best_class_indices[f]], best_class_probabilities[f])

		return input_data
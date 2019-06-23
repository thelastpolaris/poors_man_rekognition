from progress.bar import Bar
from sklearn.neighbors import NearestNeighbors
import os, pickle
import numpy as np
from collections import Counter
from ..kernel import Kernel
from ...utils import utils
import abc
import facenet.src.facenet as facenet

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class FaceRecognizerKernel(Kernel):
	def __init__(self):
		super().__init__()

	@abc.abstractmethod
	def load_model(self):
		pass

	@abc.abstractmethod
	def calculate_embeddings(self, faces, data_from_pipeline=True, batch_size=100, image_size=160):
		pass

	def train(self, dataset_folder, model_name):
		self.load_model()

		dataset = facenet.get_dataset(dataset_folder)

		# Check that there are at least one training image per class
		# for cls in dataset:
		# assert(len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

		paths, labels = facenet.get_image_paths_and_labels(dataset)

		print('Number of classes: %d' % len(dataset))
		print('Number of images: %d' % len(paths))

		print("Calculating embeddings for new data")
		data_emb = self.calculate_embeddings(paths, False)

		class_names = [cls.name.replace('_', ' ') for cls in dataset]

		# Saving classifier model
		with open(model_name, 'wb') as outfile:
			pickle.dump((data_emb, class_names, labels), outfile)
		print('Saved classifier model to file "%s"' % model_name)

	def predict(self, connection, frames_face_boxes, frames_reader):
		print("Recognizing the faces")
		self.load_model()

		infile = open(self._facenet_classifier, 'rb')
		(model_emb, class_names, labels) = pickle.load(infile)
		print('Loaded classifier model from file "%s"' % self._facenet_classifier)

		n_ngbr = 10
		nbrs = NearestNeighbors(n_neighbors=n_ngbr, algorithm='ball_tree').fit(model_emb)

		bar = Bar('Processing', max = frames_reader.frames_num())

		faces_names = []
		i = 0

		frames_generator = frames_reader.get_frames()

		for frames_data, frames_pts in frames_generator:
			boxes = frames_face_boxes[i]
			i += 1

			frame_names = []

			if len(boxes):
				faces = utils.extract_boxes(frames_data, boxes)

				emb_array = self.calculate_embeddings(faces)

				if len(faces):
					distances, indices = nbrs.kneighbors(emb_array)

					for f in range(len(faces)):
						inds = indices[f]
						classes = np.array([labels[i] for i in inds])
						label = Counter(classes).most_common(1)[0][0]

						person_name = class_names[label]
						confidence = np.sum(classes == label) / n_ngbr

						if confidence <= 0.3:
							person_name = "Unknown"

						frame_names.append((person_name, confidence))

			bar.next()

			faces_names.append(frame_names)

		connection.send(faces_names)

		bar.finish()
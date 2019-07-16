from progress.bar import Bar
from sklearn.neighbors import NearestNeighbors
import os, pickle
import numpy as np
from ..kernel import Kernel
from ...utils import utils
import abc
import facenet.src.facenet as facenet
import time
# import faiss
from collections import Counter
import math
import cv2
import scipy

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class FaceRecognizerKernel(Kernel):
	def __init__(self):
		self._classifier = None
		super().__init__()
		self._normalize_image = True
		self._preprocess = True
		self._preprocess_batch = False
		self._margin = 0
		self._embedding_size = 512

	@abc.abstractmethod
	def load_model(self):
		pass

	@abc.abstractmethod
	def preprocess_face(self, face_img):
		pass

	@abc.abstractmethod
	def calculate_embeddings(self, face_img):
		pass

	def preprocess_batch(self, faces):
		imgs = []

		for face in faces:
			if self._preprocess:
				imgs.append(self.preprocess_face(face))

		return np.float32(imgs)

	def process_faces(self, faces, data_from_pipeline=True, batch_size=100, image_size=160):
		if data_from_pipeline:
			emb_array = []

			if len(faces):
				emb_array = self.calculate_embeddings(self.preprocess_batch(faces))
		else:
			num_images = len(faces)
			num_batches_per_epoch = int(math.ceil(1.0 * num_images / batch_size))
			emb_array = np.zeros((num_images, self._embedding_size))
			bar = Bar('Processing', max= num_images)

			for i in range(num_batches_per_epoch):
				start_index = i * batch_size
				end_index = min((i + 1) * batch_size, num_images)

				paths_batch = faces[start_index:end_index]
				images = facenet.load_data(paths_batch, False, False, image_size, self._normalize_image)

				if not self._normalize_image:
					# Strip off the fractional part
					images = images.astype(np.int)
					# back to float32
					images = images.astype(np.float32)

				if self._preprocess_batch:
					images = self.preprocess_batch(images)

				emb_array[start_index:end_index, :] = self.calculate_embeddings(images)
				bar.next(end_index - start_index)

		return emb_array

	def train(self, dataset_folder, model_name, batch_size = 100, backend="FAISS"):
		self.load_model()

		dataset = facenet.get_dataset(dataset_folder)

		# Check that there are at least one training image per class
		# for cls in dataset:
		# assert(len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

		paths, labels = facenet.get_image_paths_and_labels(dataset)

		print('Number of classes: %d' % len(dataset))
		print('Number of images: %d' % len(paths))

		print("Calculating embeddings for new data")
		data_emb = self.process_faces(paths, data_from_pipeline=False, batch_size = batch_size )
		data_emb = np.float32(data_emb)
		dimension = int(data_emb.shape[1])

		class_names = [cls.name.replace('_', ' ') for cls in dataset]

		if backend == "FAISS":
			nlist = int(math.sqrt(data_emb.shape[0]))  # number of clusters
			quantiser = faiss.IndexFlatL2(dimension)
			index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_L2)

			index.train(data_emb)
			index.add(data_emb)

			# Saving classifier model
			faiss.write_index(index, model_name)  # save the index to disk

			with open(model_name + ".names", 'wb') as outfile:
				pickle.dump((class_names, labels), outfile)
		else:
			with open(model_name, 'wb') as outfile:
				pickle.dump((data_emb, class_names, labels), outfile)
		print('\nSaved classifier model to file "%s"' % model_name)

	def predict(self, connection, frames_face_boxes, frames_reader, benchmark: bool, backend="FAISS", n_ngbr = 10,
				tracked_faces = None, distance_threshold = 0.5, serialize_dir = ""):
		print("Recognizing the faces")
		face_embs_file = os.path.join(serialize_dir, "frames_face_embs.pkl")

		ser_faces_embs = []
		if serialize_dir and os.path.isfile(face_embs_file):
			with open(os.path.join(serialize_dir, face_embs_file), 'rb') as f:
				ser_faces_embs = pickle.load(f)
		else:
			self.load_model()

		benchmark_data = None
		if benchmark:
			benchmark_data = dict()

		if backend is "SciKit":
			infile = open(self._classifier, 'rb')
			(model_emb, class_names, labels) = pickle.load(infile)
			nbrs = NearestNeighbors(n_neighbors=n_ngbr, algorithm='auto', metric='cosine').fit(model_emb)
		else: # FAISS is default
			infile = open(self._classifier + ".names", 'rb')
			(class_names, labels) = pickle.load(infile)
			index = faiss.read_index(self._classifier)  # load the index

		print('Loaded classifier model from file "%s"' % self._classifier)

		faces_names = []
		faces_embs = []

		frames_generator = frames_reader.get_frames()

		if benchmark:
			start = time.time()

		bar = Bar('Processing', max=frames_reader.frames_num())

		for i, (frames_data, frames_pts) in enumerate(frames_generator):
			boxes = frames_face_boxes[i]

			frame_names = []
			frame_embs = []

			if len(boxes):
				faces = utils.extract_boxes(frames_data, boxes, self._margin)

				if serialize_dir and ser_faces_embs:
					emb_array = ser_faces_embs[i]
				else:
					emb_array = self.process_faces(faces)

				if len(faces):
					if backend is "SciKit":
						distances, indices = nbrs.kneighbors(emb_array)
					else:
						distances, indices = index.search(emb_array, n_ngbr)

					for f in range(len(faces)):
						classes = []

						# Update names according to distance threshold
						for count, index in enumerate(indices[f]):
							l = labels[index]
							distance = distances[f][count]
							if distance_threshold:
								if distance > distance_threshold:
									l = -1

							classes.append(l)

						classes = np.array(classes)
						label = Counter(classes).most_common(1)[0][0]

						if label != -1:
							person_name = class_names[label]
						else:
							person_name = "Unknown"

						confidence = np.sum(classes == label) / n_ngbr

						frame_names.append((person_name, confidence))
						frame_embs.append(emb_array[f])
			bar.next()
			faces_names.append(frame_names)
			faces_embs.append(frame_embs)

		if serialize_dir and not ser_faces_embs:
			with open(face_embs_file, 'wb') as f:
				pickle.dump(faces_embs, f)

		if tracked_faces:
			for person in tracked_faces:
				p_names = []

				for p_frames in person:
					frame_num = p_frames[0]
					face_num = p_frames[1]
					p_names.append(faces_names[frame_num][face_num][0])

				# Get person by majority voting. Confidence here is number of most common name/all names
				most_common_p = Counter(p_names).most_common()[0]
				most_common_p = (most_common_p[0], most_common_p[1]/len(p_names))

				# Adjust according to detected persons
				for p_frames in person:
					faces_names[p_frames[0]][p_frames[1]] = most_common_p

		if benchmark:
			end = time.time()
			benchmark_data["Recognition Time"] = end - start

		connection.send((faces_names, benchmark_data))

		bar.finish()
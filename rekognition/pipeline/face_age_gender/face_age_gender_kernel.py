from progress.bar import Bar
import os, pickle
import numpy as np
from ..kernel import Kernel
from ...utils import utils
import abc
import time

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class FaceAgeGenderKernel(Kernel):
	def __init__(self):
		super().__init__()
		self._classifier = None
		self._normalize_image = True
		self._preprocess = True
		self._preprocess_batch = False

	@abc.abstractmethod
	def load_model(self):
		pass

	def predict(self, connection, frames_face_boxes, frames_reader, benchmark: bool, backend="FAISS", n_ngbr = 10,
				face_tracking = True, distance_threshold = 0.5):
		print("Detecting age and gender")
		self.load_model()

		benchmark_data = None
		if benchmark:
			benchmark_data = dict()

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
				faces = utils.extract_boxes(frames_data, boxes)

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
							if distances[f][count] > distance_threshold:
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

		# Detect persons
		if face_tracking and frames_reader.content_type == "video":
			persons = [[(0, i)] for i in range(len(frames_face_boxes[0]))]
			persons_frames = [[i, face_box, False] for i, face_box in enumerate(frames_face_boxes[0])]

			for i, face_boxes in enumerate(frames_face_boxes):
				if i == 0:
					continue

				for b, box in enumerate(face_boxes):
					found = False
					for p, person_f in enumerate(persons_frames):
						if utils.IoU(person_f[1], box) > utils.IOU_THRESHOLD:
							persons[person_f[0]].append((i, b))
							persons_frames[p][1] = box
							persons_frames[p][2] = True
							found = True
							break

					if not found:
						persons.append([(i, b)])
						persons_frames.append([len(persons) - 1, box, True])

				for p, person_f in enumerate(persons_frames):
					if person_f[2]:
						persons_frames[p][2] = False
					else:
						del persons_frames[p]

			for person in persons:
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
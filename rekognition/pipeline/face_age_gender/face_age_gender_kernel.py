from progress.bar import Bar
from ..kernel import Kernel
from ...utils import utils
import abc
import time
import os
from collections import Counter
import numpy as np

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class FaceAgeGenderKernel(Kernel):
	def __init__(self):
		self._classifier = None
		super().__init__()
		self._normalize_image = True
		self._preprocess = True
		self._preprocess_batch = False
		self._margin = 0

	@abc.abstractmethod
	def load_model(self):
		pass

	@abc.abstractmethod
	def preprocess(self, faces_img):
		pass

	@abc.abstractmethod
	def get_age_gender(self, faces_img):
		pass

	def predict(self, connection, frames_face_boxes, frames_reader, benchmark: bool, tracked_faces = None):
		print("Detecting Age and Gender")
		self.load_model()

		benchmark_data = None
		if benchmark:
			benchmark_data = dict()

		frames_generator = frames_reader.get_frames()

		if benchmark:
			start = time.time()

		bar = Bar('Processing', max=frames_reader.frames_num())

		faces_age = []
		faces_gender = []

		for i, (frames_data, frames_pts) in enumerate(frames_generator):
			boxes = frames_face_boxes[i]
			age = []
			gender = []

			if len(boxes):
				faces = utils.extract_boxes(frames_data, boxes, self._margin)
				age, gender = self.get_age_gender(self.preprocess(faces))

			faces_age.append(age)
			faces_gender.append(gender)

			bar.next()

		if tracked_faces:
			for person in tracked_faces:
				p_age = []
				p_gender = []

				for p_frames in person:
					frame_num = p_frames[0]
					face_num = p_frames[1]
					p_age.append(faces_age[frame_num][face_num])
					p_gender.append(faces_gender[frame_num][face_num])

				# Get gender by majority voting. Confidence here is number of most common name/all names
				most_common_age = int(np.ceil(np.mean(np.array(p_age))))
				most_common_gender = Counter(p_gender).most_common()[0][0]

				# Adjust according to detected persons
				for p_frames in person:
					faces_age[p_frames[0]][p_frames[1]] = most_common_age
					faces_gender[p_frames[0]][p_frames[1]] = most_common_gender

		if benchmark:
			end = time.time()
			benchmark_data["Recognition Time"] = end - start

		connection.send((faces_age, faces_gender, benchmark_data))

		bar.finish()
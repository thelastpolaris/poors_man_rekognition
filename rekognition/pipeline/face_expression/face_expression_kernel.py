from progress.bar import Bar
from ..kernel import Kernel
from ...utils import utils
import abc
import time
import os
from collections import Counter

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class FaceExpressionKernel(Kernel):
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
	def get_expression(self, faces_img):
		pass

	def predict(self, connection, frames_face_boxes, frames_reader, benchmark: bool,  tracked_faces = None, smoothing_window = 3):
		print("Recognizing Facial Expressions")
		self.load_model()

		benchmark_data = None
		if benchmark:
			benchmark_data = dict()

		frames_generator = frames_reader.get_frames()

		if benchmark:
			start = time.time()

		bar = Bar('Processing', max=frames_reader.frames_num())

		faces_expressions = []

		for i, (frames_data, frames_pts) in enumerate(frames_generator):
			expressions = []
			boxes = frames_face_boxes[i]

			if len(boxes):
				faces = utils.extract_boxes(frames_data, boxes, self._margin)
				expressions = self.get_expression(self.preprocess(faces))

			faces_expressions.append(expressions)

			bar.next()

		if tracked_faces:
			for person in tracked_faces:
				p_expression = []

				person_window = utils.chunks(person, 3)

				for p in person_window:
					for p_frames in p:
						frame_num = p_frames[0]
						face_num = p_frames[1]
						p_expression.append(faces_expressions[frame_num][face_num])

					# Get gender by majority voting. Confidence here is number of most common name/all names
					most_common_exp = Counter(p_expression).most_common()[0][0]

					# Adjust according to detected persons
					for p_frames in p:
						faces_expressions[p_frames[0]][p_frames[1]] = most_common_exp

		if benchmark:
			end = time.time()
			benchmark_data["Recognition Time"] = end - start

		connection.send((faces_expressions, benchmark_data))

		bar.finish()
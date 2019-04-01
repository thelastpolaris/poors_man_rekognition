import av, abc
from progress.bar import Bar
from rekognition.pipeline.pipeline_element import PipelineElement
import numpy as np

class Person:
	__predicted_name = None
	__prediction_prob = None

	def __init__(self, predicted_name, prediction_prob):
		self.__predicted_name = predicted_name
		self.__prediction_prob = prediction_prob

	def get_predicted_name(self):
		return self.__predicted_name

	def get_JSON(self):
		person = {}

		person["name"] = self.__predicted_name
		person["probability"] = self.__prediction_prob

		return person

class Face:
	__bounding_box = None
	__face_image = None
	__person = None

	def get_face_image(self):
		return self.__face_image

	def get_bounding_box(self):
		return self.__bounding_box

	def get_person(self):
		return self.__person

	def set_person(self, predicted_name, prediction_prob):
		self.__person = Person(predicted_name, prediction_prob)

	def __init__(self, face_image, bounding_box):
		self.__face_image = face_image
		self.__bounding_box = bounding_box

	def get_JSON(self):
		face = {}

		bb = self.__bounding_box
		face["bounding_box"] = {"left": float(bb[0]), "right": float(bb[1]), "top": float(bb[2]), "bottom": float(bb[3])}

		face["person"] = None
		if self.__person:
			face["person"] = self.__person.get_JSON()

		return face


# can be either image or video frame
class Data:
	__image_data = None
	__faces = None

	def __init__(self, image_data):
		self.__image_data = image_data
		self.__faces = []

	def add_face(self, face_image, bounding_box):
		face = Face(face_image, bounding_box)
		self.__faces.append(face)

	def get_faces(self):
		return self.__faces

	def get_image_data(self, delete_data = False):
		if type(self.__image_data) == np.ndarray:
			image_data = self.__image_data

			if delete_data:
				self.__image_data = None

			return image_data
	
	def get_JSON(self):
		data = {}

		faces = []

		for face in self.__faces:
			f = dict()

			f["face"] = face.get_JSON()

			# f["bounding_box"]
			faces.append(f)

		data["faces"] = faces
		return data


class DataHandlerElem(PipelineElement):
	def __init__(self):
		_num_of_images = 0

	@property
	def num_of_images(self, num_of_images):
		return self._num_of_images
	
	@num_of_images.setter
	def num_of_images(self, num_of_images):
		self._num_of_images = num_of_images
		self.parent_pipeline.num_of_images = num_of_images

	def run(self, path_to_file):
		self.parent_pipeline.path_to_file = path_to_file
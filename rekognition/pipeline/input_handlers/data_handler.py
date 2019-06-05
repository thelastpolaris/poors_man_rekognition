import av, abc
from progress.bar import Bar
from ..pipeline_element import PipelineElement
import numpy as np

class Person:
	def __init__(self, predicted_name, prediction_prob):
		self._predicted_name = predicted_name
		self._prediction_prob = prediction_prob

	@property
	def predicted_name(self):
		return self._predicted_name

	def get_JSON(self):
		person = {}

		person["name"] = self._predicted_name
		person["confidence"] = self._prediction_prob

		return person

class Face:
	@property
	def face_image(self):
		return self._face_image
	
	@property
	def bounding_box(self):
		return self._bounding_box

	@property
	def person(self):
		return self._person
	
	def set_person(self, predicted_name, prediction_prob):
		self._person = Person(predicted_name, prediction_prob)

	def __init__(self, face_image, bounding_box):
		self._person = None
		self._face_image = face_image
		self._bounding_box = bounding_box

	def get_JSON(self):
		face = {}

		bb = self._bounding_box
		face["bounding_box"] = {"left": float(bb[0]), "right": float(bb[1]), "top": float(bb[2]), "bottom": float(bb[3])}

		face["person"] = None
		if self._person:
			face["person"] = self._person.get_JSON()

		return face

class DataHandlerElem(PipelineElement):
	def __init__(self):
		super().__init__()

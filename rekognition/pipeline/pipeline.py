class Data:
	def __init__(self):
		self._frames_reader = None
		self._frames_pts = None
		self._frames_face_boxes = None
		self._frames_face_names = None

	@property
	def frames_reader(self):
		return self._frames_reader

	@frames_reader.setter
	def frames_reader(self, frames_reader):
		self._frames_reader = frames_reader


class Pipeline:
	def __init__(self, elements):
		self._elements = []
		for elem in elements:
			self.add_elements(elem)

		self._results = []
		self._num_of_images = 0
		self._data_holder = Data()

	@property
	def num_of_images(self):
		return self._num_of_images

	@num_of_images.setter
	def num_of_images(self, num_of_images):
		self._num_of_images = num_of_images

	def add_elements(self, element):
		# Check for whether we can put elements in a pipeline
		self._elements.append(element)
		element.parent_pipeline = self

	def run(self, params_dict):
		# Self._results will be cleared in the DataHandler
		assert (len(self._elements)), "Pipeline needs to have at least one PipelineElement"

		for elem in self._elements:
			if elem in params_dict.keys():
				elem.run(self._data_holder, **params_dict[elem])
			else:
				elem.run(self._data_holder)

		return True

	def __str__(self):
		output = ""

		elems_len = len(self._elements)

		for i in range(0, elems_len):
			elem = self._elements[i]
			output += elem.__str__()
			
			if i != elems_len - 1:
				output += "-->"

		return output


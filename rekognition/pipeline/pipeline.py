class Pipeline:
	def __init__(self):
		self._elements = []
		self._results = []
		self._num_of_images = 0
		self._path_to_file = None

	@property
	def filename(self):
		split_filename = self._path_to_file.split("/")
		# Delete empty arrays
		split_filename = [x for x in split_filename if x != '']
		return split_filename[len(split_filename) -1 ]

	@property
	def path_to_file(self):
		return self._path_to_file

	@path_to_file.setter
	def path_to_file(self, path_to_file):
		self._path_to_file = path_to_file

	@property
	def num_of_images(self):
		return self._num_of_images

	@num_of_images.setter
	def num_of_images(self, num_of_images):
		self._num_of_images = num_of_images

	def add_element(self, pipeline_elem, input_data):
		# Check for whether we can put elements in a pipeline
		self._elements.append([pipeline_elem, input_data])
		pipeline_elem.parent_pipeline = self

	def run(self):
		self._results = dict()

		for elem in self._elements:
			data = elem[1]

			if data in self._results.keys():
				data = self._results[data]

			res = elem[0].run(data)
			# Save results of element work
			self._results[elem[0]] = res

	def __str__(self):
		output = ""

		elems_len = len(self._elements)

		for i in range(0, elems_len):
			elem = self._elements[i]
			output += str(elem[0].__class__.__name__)
			
			if i != elems_len - 1:
				output += "-->"

		return output


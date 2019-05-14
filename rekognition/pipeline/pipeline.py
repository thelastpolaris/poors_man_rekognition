from multiprocessing import Process, Manager

class Pipeline:
	def __init__(self, elements):
		self._elements = []
		for elem in elements:
			self.add_elements(elem)

		self._results = []
		self._num_of_images = 0
		# self._path_to_file = None

	# @property
	# def filename(self):
	# 	split_filename = self._path_to_file.split("/")
	# 	# Delete empty arrays
	# 	split_filename = [x for x in split_filename if x != '']
	# 	return split_filename[len(split_filename) -1 ]
	#
	# @property
	# def path_to_file(self):
	# 	return self._path_to_file
	#
	# @path_to_file.setter
	# def path_to_file(self, path_to_file):
	# 	self._path_to_file = path_to_file

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

	def run(self):
		# Self._results will be cleared in the DataHandler
		manager = Manager()
		ns = manager.Namespace()
		self._results = []
		ns.input_data = []
		results = Manager().list(self._results)

		for elem in self._elements:
			p = Process(target=elem.run, args=(results,))
			p.start()
			p.join()
			frame = results[0]
			# elem.run(self._results)

		return True

	def __str__(self):
		output = ""

		elems_len = len(self._elements)

		for i in range(0, elems_len):
			elem = self._elements[i]
			output += str(elem.__class__.__name__)
			
			if i != elems_len - 1:
				output += "-->"

		return output


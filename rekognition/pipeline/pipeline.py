class Pipeline:
	__elements = None
	__results = None
	__num_of_images = None
	__path_to_file = None

	def __init__(self):
		self.__elements = []
		self.__results = []

	def set_num_of_images(self, num_of_images):
		self.__num_of_images = num_of_images

	def get_num_of_images(self):
		return self.__num_of_images

	def set_path_to_file(self, name_of_file):
		self.__name_of_file = name_of_file

	def get_path_to_file(self):
		return self.__name_of_file

	def get_filename(self):
		split_filename = self.get_path_to_file().split("/")
		return split_filename[len(split_filename) -1 ]

	def add_element(self, pipeline_elem, input_data):
		# Check for whether we can put elements in a pipeline
		self.__elements.append([pipeline_elem, input_data])
		pipeline_elem.set_parent_pipeline(self)

	def run(self):
		self.__results = dict()

		for elem in self.__elements:
			data = elem[1]

			if data in self.__results.keys():
				data = self.__results[data]

			res = elem[0].run(data)
			# Save results of element work
			self.__results[elem[0]] = res

	def __str__(self):
		output = ""

		elems_len = len(self.__elements)

		for i in range(0, elems_len):
			elem = self.__elements[i]
			output += str(elem[0].__class__.__name__)
			
			if i != elems_len - 1:
				output += "-->"

		return output


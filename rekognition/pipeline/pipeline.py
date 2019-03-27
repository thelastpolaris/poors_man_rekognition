class Pipeline:
	__elements = []
	__results = []

	def __init__(self):
		pass
		# self.name = name

	def add_element(self, pipeline_elem, input_data):
		self.__elements.append([pipeline_elem, input_data])

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


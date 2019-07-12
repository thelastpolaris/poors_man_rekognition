import collections
from rekognition.pipeline.pipeline_element import PipelineElement

class Data:
	def __init__(self):
		self.__values = collections.OrderedDict()
		self.__benchmark = self.Benchmark()

	def add_value(self, value_name, value):
		if value_name in self.__values.keys():
			print("{} exists. New value won't be added. Use update_value()".format(value_name))
			return False
		else:
			self.__values[value_name] = value
			return True

	def update_value(self, value_name, new_value):
		if value_name in self.__values.keys():
			self.__values[value_name] = new_value
			return True
		else:
			return False

	def get_value(self, value_name):
		if value_name in self.__values.keys():
			return self.__values[value_name]
		else:
			return None

	@property
	def benchmark(self):
		return self.__benchmark

	class Benchmark:
		def __init__(self):
			self.__elem_values = collections.OrderedDict()

		def add_value(self, element: PipelineElement, value_name: str, value):
			if element not in self.__elem_values.keys():
				self.__elem_values[element] = {}

			self.__elem_values[element][value_name] = value

		def __str__(self):
			benchmark_out = ""
			for k, v in self.__elem_values.items():
				newline = "\n" if benchmark_out else ""
				benchmark_out = "{}{} {} {}".format(benchmark_out, newline, k, v)

			return benchmark_out

		def save_benchmark(self, path_to_file):
			with open(path_to_file + ".txt", 'w') as data:
				data.write(self.__str__())
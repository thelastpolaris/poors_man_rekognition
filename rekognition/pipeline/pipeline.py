import time
import collections
from rekognition.pipeline.pipeline_element import PipelineElement

class Data:
	def __init__(self):
		self._frames_correlation = None
		self._frames_group = None

		self._frames_reader = None
		self._frames_pts = None
		self._frames_face_boxes = None
		self._frames_face_names = None
		self._frames_face_embs = None

		self.__benchmark = self.Benchmark()

	@property
	def frames_reader(self):
		return self._frames_reader

	@frames_reader.setter
	def frames_reader(self, frames_reader):
		self._frames_reader = frames_reader

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

		def print_benchmark(self):
			for k, v in self.__elem_values.items():
				print(k, v)

		def save_benchmark(self, path_to_file):
			benchmark_out = ""
			for k, v in self.__elem_values.items():
				newline = "\n" if benchmark_out else ""
				benchmark_out = "{}{} {} {}".format(benchmark_out, newline, k, v)

			with open(path_to_file + ".txt", 'w') as data:
				data.write(benchmark_out)
			# "output/" + output_name + '_output.mp4'

class Pipeline:
	def __init__(self, elements):
		self.__elements = []
		self.__data_holder = None

		for elem in elements:
			self.add_elements(elem)

	def add_elements(self, element):
		# Check for whether we can put elements in a pipeline
		self.__elements.append(element)
		element.parent_pipeline = self

	def run(self, params_dict, benchmark = False):
		assert (len(self.__elements)), "Pipeline needs to have at least one PipelineElement"

		start = time.time()

		self.__data_holder = Data()

		for elem in self.__elements:
			if elem in params_dict.keys() and elem != self:
				elem.run(self.__data_holder, benchmark = benchmark, **params_dict[elem])
			else:
				elem.run(self.__data_holder, benchmark = benchmark)

		end = time.time()

		self.__data_holder.benchmark.print_benchmark()
		print("Done! Total time elapsed {:.2f} seconds".format(end - start))
		if self in params_dict.keys():
			if "out_name" in params_dict[self]:
				self.__data_holder.benchmark.save_benchmark(params_dict[self]["out_name"])

		return True

	def __str__(self):
		output = ""

		elems_len = len(self.__elements)

		for i in range(0, elems_len):
			elem = self.__elements[i]
			output += elem.__str__()
			
			if i != elems_len - 1:
				output += "-->"

		return output


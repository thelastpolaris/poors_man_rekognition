import time
import collections
from rekognition.pipeline.pipeline_element import PipelineElement
import os
import json

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

		return input_data

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

class Pipeline:
	def __init__(self, elements):
		self.__elements = []
		self.__data_holder = None

		for elem in elements:
			self.add_elements(elem)

	class WrongPipelineOrder(Exception):
		"""Raised when pipeline doesn't contain an element required by a new PipelineElement"""
		pass

	def add_elements(self, element):
		# Check for whether we can put elements in a pipeline
		required_elems = element.requires()
		if required_elems:
			if required_elems is not list:
				required_elems = [required_elems]

			for req_elem in required_elems:
				if not any([type(elem) == req_elem for elem in self.__elements]):
					raise self.WrongPipelineOrder

		self.__elements.append(element)

		element.parent_pipeline = self

	def run(self, params_dict, benchmark = False, save_JSON = True):
		assert (len(self.__elements)), "Pipeline needs to have at least one PipelineElement"

		start = time.time()

		self.__data_holder = Data()

		for elem in self.__elements:
			if elem in params_dict.keys() and elem != self:
				elem.run(self.__data_holder, benchmark = benchmark, **params_dict[elem])
			else:
				elem.run(self.__data_holder, benchmark = benchmark)

		end = time.time()
		print("Done! Total time elapsed {:.2f} seconds".format(end - start))

		if save_JSON:
			self.save_JSON(params_dict)

		if benchmark:
			print(self.__data_holder.benchmark)

			if self in params_dict.keys():
				if "out_name" in params_dict[self]:
					self.__data_holder.benchmark.save_benchmark(params_dict[self]["out_name"])

		return True

	def save_JSON(self, params_dict):
		json_objects = []

		for elem in self.__elements:
			json_objects = elem.get_JSON(self.__data_holder, json_objects)

		filename = "{}.{}".format(params_dict[self]["out_name"], "json")

		with open(filename, "w") as write_file:
			json.dump(json_objects, write_file, indent=4)

	def __str__(self):
		output = ""

		elems_len = len(self.__elements)

		for i in range(0, elems_len):
			elem = self.__elements[i]
			output += elem.__str__()
			
			if i != elems_len - 1:
				output += "-->"

		return output


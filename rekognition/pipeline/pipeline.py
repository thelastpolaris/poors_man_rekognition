from .data import Data
import time
import json

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
		# Check for whether we can put element in a pipeline
		required_elems = element.requires()

		if required_elems:
			found = False

			f = lambda test_elem: any([type(elem) == test_elem for elem in self.__elements])

			if type(required_elems) != tuple:
				found = f(required_elems) # handle single elements
			else:
				for req_elem in required_elems:
					found = f(req_elem)
					if found:
						break

			if not found:
				raise self.WrongPipelineOrder(element)

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


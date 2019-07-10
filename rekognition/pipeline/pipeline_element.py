from abc import ABC, abstractmethod

# Abstract class
class PipelineElement(ABC):
	def __init__(self, kernel = None):
		self.__parent_pipeline = None
		self.__kernel = kernel

	@property
	def parent_pipeline(self):
		return self.__parent_pipeline

	@parent_pipeline.setter
	def parent_pipeline(self, parent_pipeline):
		self.__parent_pipeline = parent_pipeline

	@property
	def kernel(self):
		return self.__kernel

	@kernel.setter
	def kernel(self, kernel):
		self.__kernel = kernel

	# Pure virtual function
	@abstractmethod
	def run(self, **args):
		pass

	@abstractmethod
	def requires(self):
		pass

	def get_JSON(self, data, json_objects):
		return json_objects

	def benchmark(self, **args):
		pass

	def __str__(self):
		if self.__kernel:
			return str(self.__class__.__name__) + "({})".format(str(self.__kernel.__class__.__name__))
		else:
			return str(self.__class__.__name__)
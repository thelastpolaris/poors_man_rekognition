from abc import ABC, abstractmethod

# Abstract class
class PipelineElement(ABC):
	def __init__(self, kernel = None):
		self._parent_pipeline = None
		self._kernel = kernel

	@property
	def parent_pipeline(self):
		return self._parent_pipeline

	@parent_pipeline.setter
	def parent_pipeline(self, parent_pipeline):
		self._parent_pipeline = parent_pipeline

	@property
	def kernel(self):
		return self._kernel

	@kernel.setter
	def kernel(self, kernel):
		self._kernel = kernel

	# Pure virtual function
	@abstractmethod
	def run(self, **args):
		pass

	def __str__(self):
		if self._kernel:
			return str(self.__class__.__name__) + "({})".format(str(self._kernel.__class__.__name__))
		else:
			return str(self.__class__.__name__)
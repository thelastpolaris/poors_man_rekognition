import abc
from .kernel import Kernel

# Abstract class
class PipelineElement:
	def __init__(self):
		self._parent_pipeline = None
		self._kernel = None

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
	@abc.abstractmethod
	def run(self, input_data):
		pass
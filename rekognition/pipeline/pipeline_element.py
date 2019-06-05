from abc import ABC, abstractmethod

# Abstract class
class PipelineElement(ABC):
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
	@abstractmethod
	def run(self, **args):
		pass
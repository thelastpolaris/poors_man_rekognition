import abc

# Abstract class
class PipelineElement:

	def __init__(self):
		_parent_pipeline = None

	@property
	def parent_pipeline(self):
		return self._parent_pipeline

	@parent_pipeline.setter
	def parent_pipeline(self, parent_pipeline):
		self._parent_pipeline = parent_pipeline

	# Pure virtual function
	@abc.abstractmethod
	def run(self, input_data):
		pass
import abc

# Abstract class
class PipelineElement:
	__parent_pipeline = None

	def __init__(self):
		pass

	def set_parent_pipeline(self, pipeline):
		self.__parent_pipeline = pipeline

	def get_parent_pipeline(self):
		return self.__parent_pipeline

	# Pure virtual function
	@abc.abstractmethod
	def run(self, input_data):
		pass
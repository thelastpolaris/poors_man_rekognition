import abc

# Abstract class
class PipelineElement:
	def __init__(self):
		pass

	# Pure virtual function
	@abc.abstractmethod
	def run(input_data):
		pass
from ..pipeline_element import PipelineElement

class OutputHandler(PipelineElement):
	def __init__(self, output_name):
		self._output_name = output_name

	@property
	def output_name(self):
		return self._output_name

	@output_name.setter
	def output_name(self, output_name):
		self._output_name = output_name

	def run(self, path_to_data):
		pass
import av
from progress.bar import Bar
import abc
from ..pipeline_element import PipelineElement

class FaceRecognizerElem(PipelineElement):
	# Pure virtual function
	@abc.abstractmethod
	def train():
		pass

	def run(self, path_to_video):
		pass
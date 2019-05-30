import av
from progress.bar import Bar
import abc
from ..pipeline_element import PipelineElement

class FaceRecognizerElem(PipelineElement):
	def __init__(self, kernel):
		self.kernel = kernel

	def run(self, data):
		results = self.kernel.run((data._frames_faces, ))
		# print(results)
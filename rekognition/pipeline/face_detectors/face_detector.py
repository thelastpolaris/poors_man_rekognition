import av
from progress.bar import Bar
from ..pipeline_element import PipelineElement

class FaceDetectorElem(PipelineElement):
	def __init__(self, kernel):
		self.kernel = kernel

	def run(self, data):
		data._frames_face_boxes, data._frames_pts = self.kernel.run((data.frames_reader, ))

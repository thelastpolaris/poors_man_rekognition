from ..pipeline_element import PipelineElement

class FaceRecognizerElem(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data):
		data._frames_face_names = self.kernel.run(data._frames_face_boxes, data.frames_reader)

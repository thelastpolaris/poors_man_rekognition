from ..pipeline_element import PipelineElement

class FaceDetectorElem(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, min_score = 0.7):
		data._frames_face_boxes, data._frames_pts = self.kernel.run(data.frames_reader, min_score)

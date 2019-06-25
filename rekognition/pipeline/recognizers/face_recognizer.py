from ..pipeline_element import PipelineElement

class FaceRecognizerElem(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark = False, backend="FAISS"):
		data._frames_face_names, benchmark_data = self.kernel.run(data._frames_face_boxes, data.frames_reader, benchmark, backend)

		if benchmark:
			self.benchmark(data, benchmark_data)

	def benchmark(self, data, benchmark_data):
		for k, v in benchmark_data.items():
			if k != "scores":
				data.benchmark.add_value(self, k, v)

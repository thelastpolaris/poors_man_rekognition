from ..pipeline_element import PipelineElement
from ..face_detectors.face_detector import FaceDetectorElem
from ...utils.utils import traverse_group

class FaceExpressionRecognizer(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark = False):
		tracked_faces = data.get_value("tracked_faces")

		face_expressions, benchmark_data = \
			self.kernel.run(data.get_value("frames_face_boxes"), data.get_value("frames_reader"), benchmark, tracked_faces)
		data.add_value("frames_face_expressions", face_expressions)

		if benchmark:
			self.benchmark(data, benchmark_data)

	def requires(self):
		return FaceDetectorElem

	def get_JSON(self, data, json_holder):
		frames_group = data.get_value("frames_group")
		face_expressions = data.get_value("frames_face_expressions")
		json_objects = json_holder["frames"]

		for (i, all_count) in traverse_group(len(face_expressions), frames_group):
			face_exp = face_expressions[i]

			for f, face in enumerate(json_objects[all_count]["faces"]):
				face["expression"] = face_exp[f]

		json_holder["frames"] = json_objects
		return json_objects

	def benchmark(self, data, benchmark_data):
		for k, v in benchmark_data.items():
			if k != "scores":
				data.benchmark.add_value(self, k, v)

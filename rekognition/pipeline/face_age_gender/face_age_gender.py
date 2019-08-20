from ..pipeline_element import PipelineElement
from ..face_detectors.face_detector import FaceDetectorElem
from ...utils.utils import traverse_group

class FaceAgeGenderElem(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark = False):
		tracked_faces = data.get_value("tracked_faces")

		frames_faces_age, frames_faces_gender, benchmark_data = \
			self.kernel.run(data.get_value("frames_face_boxes"), data.get_value("frames_reader"), benchmark,
																							tracked_faces)
		data.add_value("frames_face_age", frames_faces_age)
		data.add_value("frames_faces_gender", frames_faces_gender)

		if benchmark:
			self.benchmark(data, benchmark_data)

	def requires(self):
		return FaceDetectorElem

	def get_JSON(self, data, json_holder):
		frames_group = data.get_value("frames_group")
		frames_face_age = data.get_value("frames_face_age")
		frames_faces_gender = data.get_value("frames_faces_gender")

		json_objects = json_holder["frames"]

		for (i, all_count) in traverse_group(len(frames_face_age), frames_group):
			frame_age = frames_face_age[i]
			frame_genders = frames_faces_gender[i]

			for f, face in enumerate(json_objects[all_count]["faces"]):
				face["age"] = frame_age[f]
				face["gender"] = frame_genders[f]

		json_holder["frames"] = json_objects

		return json_objects

	def benchmark(self, data, benchmark_data):
		for k, v in benchmark_data.items():
			if k != "scores":
				data.benchmark.add_value(self, k, v)

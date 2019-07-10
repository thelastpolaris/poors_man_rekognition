from ..pipeline_element import PipelineElement
from ...utils.utils import boxes_from_cvat_xml, calculate_tp_fp_fn, traverse_group
from ..input_handlers.video_handler import VideoHandlerElem

class FaceDetectorElem(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark=False, benchmark_boxes = None, min_score = 0.7):
		frames_face_boxes, frames_pts, benchmark_data = self.kernel.run(data.get_value("frames_reader"), benchmark,
																		min_score)
		data.add_value("frames_face_boxes", frames_face_boxes)
		data.add_value("frames_pts", frames_pts)

		if benchmark:
			self.benchmark(data, benchmark_data, benchmark_boxes)

	def requires(self):
		return VideoHandlerElem

	def get_JSON(self, data, json_objects):
		frames_group = data.get_value("frames_reader").frames_group
		frames_face_boxes = data.get_value("frames_face_boxes")

		for (i, all_count) in traverse_group(frames_group):
			faces = []

			for bb in frames_face_boxes[i]:
				face = dict()
				face["bounding_box"] = {"left": float(bb[0]), "right": float(bb[1]), "top": float(bb[2]),
										"bottom": float(bb[3])}
				faces.append(face)

			json_objects[all_count]["faces"] = faces

		return json_objects

	def benchmark(self, data, benchmark_data, benchmark_boxes):
		for k, v in benchmark_data.items():
			if k != "scores":
				data.benchmark.add_value(self, k, v)

		if "scores" in benchmark_data.keys():
			scores = benchmark_data["scores"]

		if benchmark_boxes != None:
			bench_boxes, bench_w, bench_h, bench_labels = boxes_from_cvat_xml(benchmark_boxes)

			if bench_boxes:
				TP = 0
				FP = 0
				FN = 0

				frames_group = data.get_value("frames_reader").frames_group

				bench_count = 0

				for (i, frame_boxes) in enumerate(data.get_value("frames_face_boxes")):
					group = 1
					if frames_group:
						group = frames_group[i]

					for a in range(group):
						bench_count += 1
						if bench_count < len(bench_boxes) - 1:
							_TP, _FP, _FN = calculate_tp_fp_fn(frame_boxes, bench_boxes[bench_count], bench_w, bench_h)
							TP += _TP
							FP += _FP
							FN += _FN

				precision = 0 if (TP+FP) == 0 else TP/(TP + FP)
				recall = 0 if (TP + FN) == 0 else TP/(TP + FN)

				# data.benchmark.add_value(self, "Accuracy", accuracy)
				data.benchmark.add_value(self, "Precision", precision)
				data.benchmark.add_value(self, "Recall", recall)

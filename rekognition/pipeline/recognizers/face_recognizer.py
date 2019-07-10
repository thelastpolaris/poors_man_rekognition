from ..pipeline_element import PipelineElement
from ...utils.utils import boxes_from_cvat_xml, calculate_tp_fp_fn, traverse_group
from ..face_detectors.face_detector import FaceDetectorElem
class FaceRecognizerElem(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark = False, benchmark_boxes=None, backend="FAISS", n_ngbr = 10, face_tracking=True):
		frames_face_names, benchmark_data = \
			self.kernel.run(data.get_value("frames_face_boxes"), data.get_value("frames_reader"), benchmark, backend,
																								n_ngbr, face_tracking)
		data.add_value("frames_face_names", frames_face_names)

		if benchmark:
			self.benchmark(data, benchmark_data, benchmark_boxes)

	def requires(self):
		return FaceDetectorElem

	def get_JSON(self, data, json_objects):
		frames_group = data.get_value("frames_reader").frames_group
		frames_face_names = data.get_value("frames_face_names")

		for (i, all_count) in traverse_group(frames_group):
			for f, face in enumerate(json_objects[all_count]["faces"]):
				frame_names = frames_face_names[i]
				face["name"] = frame_names[f]

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

				frames_group = data.get_value("frames_reader").frames_group

				bench_count = 0

				frames_face_names = data.get_value("frames_face_names")

				for (i, frame_boxes) in enumerate(data.get_value("frames_face_boxes")):
					frame_labels = [ face_name[0].replace(" ", "_") for face_name in frames_face_names[i] ]

					group = 1
					if frames_group:
						group = frames_group[i]

					for a in range(group):

						bench_count += 1
						if bench_count < len(bench_boxes) - 1:
							_TP, _FP = calculate_tp_fp_fn(frame_boxes, bench_boxes[bench_count], bench_w, bench_h,
															   frame_labels, bench_labels[bench_count])
							TP += _TP
							FP += _FP

				precision = 0 if (TP+FP) == 0 else TP/(TP + FP)

				data.benchmark.add_value(self, "Precision", precision)
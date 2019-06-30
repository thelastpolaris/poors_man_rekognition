from ..pipeline_element import PipelineElement
from ...utils.utils import boxes_from_cvat_xml, calculate_tp_fp_fn

class FaceRecognizerElem(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark = False, backend="FAISS", n_ngbr = 10, benchmark_boxes=None):
		data._frames_face_names, benchmark_data = self.kernel.run(data._frames_face_boxes, data.frames_reader, benchmark, backend, n_ngbr)

		if benchmark:
			self.benchmark(data, benchmark_data, benchmark_boxes)

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

				frames_group = data.frames_reader.frames_group

				bench_count = 0

				for (i, frame_boxes) in enumerate(data._frames_face_boxes):
					frame_labels = [ face_name[0].replace(" ", "_") for face_name in data._frames_face_names[i] ]

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
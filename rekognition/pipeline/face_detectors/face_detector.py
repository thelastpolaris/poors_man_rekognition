from ..pipeline_element import PipelineElement
from ...utils.utils import boxes_from_cvat_xml, calculate_tp_fp_fn
from sklearn.metrics import precision_recall_curve

class FaceDetectorElem(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark=False, min_score=0.7, benchmark_boxes=None):
		data._frames_face_boxes, data._frames_pts, benchmark_data = self.kernel.run(data.frames_reader,
																					min_score,
																					benchmark)
		if benchmark:
			self.benchmark(data, benchmark_data, benchmark_boxes)

	def benchmark(self, data, benchmark_data, benchmark_boxes):
		for k, v in benchmark_data.items():
			if k != "scores":
				data.benchmark.add_value(self, k, v)

		if "scores" in benchmark_data.keys():
			scores = benchmark_data["scores"]

		if benchmark_boxes != None:
			bench_boxes, bench_w, bench_h = boxes_from_cvat_xml(benchmark_boxes)

			if bench_boxes:
				TP = 0
				FP = 0
				FN = 0

				frames_group = data.frames_reader.frames_group

				bench_count = 0

				for (i, frame_boxes) in enumerate(data._frames_face_boxes):
					_frame_boxes = frame_boxes

					group = 0
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

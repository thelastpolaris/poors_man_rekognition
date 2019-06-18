from ..pipeline_element import PipelineElement
from ...utils.utils import boxes_from_cvat_xml, IoU, restore_normalization
from sklearn.metrics import precision_recall_curve

class FaceDetectorElem(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark=False, min_score=0.7, benchmark_boxes=None):
		frames_face_boxes, data._frames_pts, benchmark_data = self.kernel.run(data.frames_reader,
																					min_score,
																					benchmark)
		new_frames_face_boxes = []
		frames_group = data.frames_reader.frames_group

		if frames_group:
			for i, face_boxes in enumerate(frames_face_boxes):
				group = frames_group[i]
				for a in range(group + 1):
					new_frames_face_boxes.append(frames_face_boxes[i])
		else:
			new_frames_face_boxes = frames_face_boxes

		data._frames_face_boxes = new_frames_face_boxes

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
			prediction_labels = []

			if bench_boxes:
				Pred_num = 0
				IoU_threshold = 0.5

				TP = 0
				FP = 0
				FN = 0

				for (i, frame_boxes) in enumerate(data._frames_face_boxes):
					frame_predictions = []
					if i < len(bench_boxes) - 1:
						for bench_box in bench_boxes[i]:
							bench_found = False
							for box in frame_boxes:
								if IoU(restore_normalization(box, bench_h, bench_w), bench_box) > IoU_threshold:
									bench_found = True
									frame_predictions.append(1)
							if not bench_found:
								frame_predictions.append(0)
								FN += 1

					for box in frame_boxes:
						Pred_num += 1
						is_true = False
						if i < len(bench_boxes) - 1:
							for bench in bench_boxes[i]:
								if IoU(restore_normalization(box, bench_h, bench_w), bench) > IoU_threshold:
									TP += 1
									is_true = True
									break
							if not is_true:
								FP += 1

				# accuracy = TP/Pred_num
				precision = 0 if (TP+FP) == 0 else TP/(TP + FP)
				recall = 0 if (TP + FN) == 0 else TP/(TP + FN)

				# data.benchmark.add_value(self, "Accuracy", accuracy)
				data.benchmark.add_value(self, "Precision", precision)
				data.benchmark.add_value(self, "Recall", recall)

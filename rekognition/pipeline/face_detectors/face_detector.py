from ..pipeline_element import PipelineElement
from ...utils.utils import boxes_from_cvat_xml, IoU, restore_normalization

class FaceDetectorElem(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark=False, min_score=0.7, benchmark_boxes=None):
		data._frames_face_boxes, data._frames_pts, benchmark_time = self.kernel.run(data.frames_reader, min_score, benchmark)

		if benchmark:
			self.benchmark(data, benchmark_time, benchmark_boxes)

	def benchmark(self, data, benchmark_time, benchmark_boxes):
		for k, v in benchmark_time.items():
			data.benchmark.add_value(self, k, v)

		if benchmark_boxes != None:
			bench_boxes = boxes_from_cvat_xml(benchmark_boxes)

			if bench_boxes:
				Num = 0
				IoU_threshold = 0.5

				TP = 0
				FP = 0
				FN = 0

				for (i, frame_boxes) in enumerate(data._frames_face_boxes):
					if i < len(bench_boxes) - 1:
						for bench_box in bench_boxes[i]:
							bench_found = False
							for box in frame_boxes:
								if IoU(restore_normalization(box, 720, 1280), bench_box) > IoU_threshold:
									bench_found = True
							if not bench_found:
								FN += 1

					for box in frame_boxes:
						is_true = False
						if i < len(bench_boxes) - 1:
							for bench in bench_boxes[i]:
								if IoU(restore_normalization(box, 720, 1280), bench) > IoU_threshold:
									TP += 1
									is_true = True
							if not is_true:
								FP += 1

				accuracy = 0 if (TP+FP) == 0 else TP/(TP + FP)
				recall = 0 if (TP + FN) == 0 else TP/(TP + FN)

				data.benchmark.add_value(self, "Accuracy", accuracy)
				data.benchmark.add_value(self, "Recall", recall)

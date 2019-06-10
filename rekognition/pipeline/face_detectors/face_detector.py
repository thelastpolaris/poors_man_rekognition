from ..pipeline_element import PipelineElement
from ...utils.utils import boxes_from_cvat_xml, IoU, normalize_box

class FaceDetectorElem(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark=True, min_score=0.7, benchmark_boxes=None):
		data._frames_face_boxes, data._frames_pts, benchmark_time = self.kernel.run(data.frames_reader, min_score, benchmark)

		if benchmark:
			self.benchmark(data, benchmark_time, benchmark_boxes)

	def benchmark(self, data, benchmark_time, benchmark_boxes):
		for k, v in benchmark_time.items():
			data.benchmark.add_value(self, k, v)

		if benchmark_boxes != None:
			bench_boxes = boxes_from_cvat_xml(benchmark_boxes)

			Num = 0
			IOU_threshold = 0.5

			TP = 0
			FP = 0
			FN = 0

			if bench_boxes:
				for (i, frame_boxes) in enumerate(data._frames_face_boxes):
					for box in frame_boxes:
						# print(bench_boxes[i])
						for bench in bench_boxes[i]:
							pass
							# print()
							# print(normalize_box(box, 1280, 720), bench, IoU(box, bench))
					# for (c, bench) in enumerate(bench_boxes[i]):
					# 	print(frame_boxes, bench_box)
						# print(i, IoU(frame_boxes, normalize_box(bench_boxes[i][c], 320, 240)))

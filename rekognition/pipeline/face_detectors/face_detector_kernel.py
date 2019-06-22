import abc
from ..kernel import Kernel
from progress.bar import Bar
import time

class FaceDetectorKernel(Kernel):
	def __init__(self):
		super().__init__()
		pass

	@abc.abstractmethod
	def load_model(self):
		pass

	@abc.abstractmethod
	def inference(self, image):
		pass

	def predict(self, connection, frames_reader, min_score: float, benchmark: bool):
		self.load_model()

		benchmark_data = None
		if benchmark:
			benchmark_data = dict()

		print("Detecting faces in video")
		bar = None
		i = 0

		all_frames_pts = []
		all_frames_face_boxes = []

		frames_generator = frames_reader.get_frames()

		if benchmark:
			start = time.time()

		enum_frames = enumerate(frames_generator)

		for i, (frames_data, frames_pts) in enum_frames:
			image = frames_data

			if bar is None:
				bar = Bar('Processing', max=frames_reader.frames_num())

			scores, boxes = self.inference(image)

			frame_boxes = []

			if min_score > 0:
				for b in range(len(boxes)):
					if boxes[b][2] and boxes[b][3]: # Check that box has width and height > 0
						if scores[b] > min_score:
							frame_boxes.append(boxes[b])
			else:
				frame_boxes = boxes
				benchmark_data["scores"] = scores

			bar.next()
			i += 1

			all_frames_face_boxes.append(frame_boxes)
			all_frames_pts.append(frames_pts)


		if benchmark:
			end = time.time()
			benchmark_data["Inference Time"] = end - start

		if bar:
			bar.finish()

		connection.send((all_frames_face_boxes, all_frames_pts, benchmark_data))

		return
import multiprocessing
import abc
from ..kernel import Kernel
from progress.bar import Bar

class FaceDetectorKernel(Kernel):
	def __init__(self):
		pass

	@abc.abstractmethod
	def load_model(self):
		pass

	@abc.abstractmethod
	def inference(self, image):
		pass

	def predict(self, connection, frames_reader, min_score):
		self.load_model()

		print("Detecting faces in video")
		bar = None
		i = 0

		all_frames_pts = []
		all_frames_face_boxes = []

		frames_generator = frames_reader.get_frames()

		for frames_data, frames_pts in frames_generator:
			i += 1
			image = frames_data

			if bar is None:
				bar = Bar('Processing', max=frames_reader.frames_num)

			scores, boxes = self.inference(image)
			frame_boxes = []

			# scores = detections[0, 1, :, 0]
			# keep_idxs = scores > keep_thresh  # find keeping indexes
			# detections = detections[0, 1, keep_idxs, :]  # select detections over threshold
			# detections = detections[:, [1, 2, 3, 4, 0]]  # reorder

			for b in range(len(boxes)):
				if scores[b] > min_score:
					frame_boxes.append(boxes[b])

			bar.next()

			all_frames_face_boxes.append(frame_boxes)
			all_frames_pts.append(frames_pts)

		if bar:
			bar.finish()

		connection.send((all_frames_face_boxes, all_frames_pts))

		return
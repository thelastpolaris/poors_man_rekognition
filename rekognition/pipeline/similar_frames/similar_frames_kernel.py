from ..kernel import Kernel
import time
from progress.bar import Bar
import abc
import numpy as np

class SimilarFramesKernel(Kernel):
	def __init__(self):
		super().__init__

	@abc.abstractmethod
	def compare(self, frame1, frame2):
		pass

	def predict(self, connection, frames_reader, benchmark: bool, batch_size = 0):
		benchmark_data = None
		if benchmark:
			benchmark_data = dict()

		print("Detecting similar frames in video")
		bar = None
		i = 0

		all_frames_pts = []
		all_frames_correlation = [0]

		frames_generator = frames_reader.get_frames()

		if benchmark:
			start = time.time()

		prev_frame = np.array([])

		for frames_data, frames_pts in frames_generator:
			corr = 0

			if bar is None:
				bar = Bar('Processing', max=frames_reader.frames_num())

			if not prev_frame.any():
				prev_frame = frames_data
			else:
				corr = self.compare(prev_frame, frames_data)

			bar.next()
			i += 1

			all_frames_pts.append(frames_pts)
			all_frames_correlation.append(corr)

		if benchmark:
			end = time.time()
			benchmark_data["Inference Time"] = end - start

		if bar:
			bar.finish()

		connection.send((all_frames_pts, all_frames_correlation, benchmark_data))
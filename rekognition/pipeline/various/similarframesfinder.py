from ..pipeline_element import PipelineElement
from ..input_handlers.video_handler import VideoFrames
from ..kernel import Kernel
import time
from progress.bar import Bar
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim

from scipy.spatial import distance

class CorrelationSimilarity(Kernel):
	def __init__(self):
		super().__init__

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

		hist = False
		mode = "ssim"

		group = []

		for frames_data, frames_pts in frames_generator:
			corr = 0

			if not prev_frame.any():
				prev_frame = frames_data
			else:
				if hist:
					hist1 = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
					hist1 = cv2.normalize(hist1, hist1).flatten()

					hist2 = cv2.calcHist([frames_data], [0], None, [256], [0, 256])
					hist2 = cv2.normalize(hist2, hist2).flatten()

					if mode == "opencv_cor":
						corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
						print(" " + str(corr))
					elif mode == "chebyshev":
						corr = 1 - distance.chebyshev(hist1, hist2)
				else:
					if mode == "ssim":
						corr = ssim(prev_frame, frames_data, multichannel=True)
				# print(" " + str(corr))
				# elif mode == "match_template":
				# 	corr

				prev_frame = frames_data

			if bar is None:
				bar = Bar('Processing', max=frames_reader.frames_num())

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
		pass


class SimilarFramesFinder(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark=False, sim_threshold = 0.97):
		data._frames_pts, data._frames_correlation, benchmark_data = self.kernel.run(data.frames_reader, benchmark)

		sim_count = 0
		frames_group = []

		for i, corr in enumerate(data._frames_correlation):
			if corr > sim_threshold:
				sim_count += 1
			else:
				frames_group.append(sim_count)
				sim_count = 0

		if sim_count:
			frames_group.append(sim_count)

		data.frames_reader.frames_group = frames_group
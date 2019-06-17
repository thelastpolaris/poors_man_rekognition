from ..pipeline_element import PipelineElement
from ..kernel import Kernel
import time
from progress.bar import Bar
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim

class TTestSimilarity(Kernel):
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

		mode = "ssim"

		for frames_data, frames_pts in frames_generator:
			corr = 0

			if not prev_frame.any():
				prev_frame = frames_data
			else:
				if mode == "opencv_cor":
					hist1 = cv2.calcHist([prev_frame],[0],None,[256],[0,256])
					hist2 = cv2.calcHist([frames_data],[0],None,[256],[0,256])
					corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
				elif mode == "ssim":
					corr = ssim(prev_frame, frames_data, multichannel=True)
				elif mode == "match_template":
					corr 

				prev_frame = frames_data

			if bar is None:
				bar = Bar('Processing', max=frames_reader.frames_num)

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

	def run(self, data, benchmark=False):
		data._frames_pts, data._frames_correlation, benchmark_data = self.kernel.run(data.frames_reader, benchmark)
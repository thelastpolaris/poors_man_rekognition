from .similar_frames_kernel import SimilarFramesKernel
import cv2

from scipy.spatial import distance

class CompHist(SimilarFramesKernel):
	def __init__(self, method = "chebyshev"):
		super().__init__()
		self._methods = ["chebyshev", "opencv_cor"]
		if method not in self._methods:
			self._method = "chebyshev"
		else:
			self._method = method

	def compare(self, frame1, frame2):
		hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
		hist1 = cv2.normalize(hist1, hist1).flatten()

		hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
		hist2 = cv2.normalize(hist2, hist2).flatten()

		corr = 0

		if self._method == "opencv_cor":
			corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
		elif self._method == "chebyshev":
			corr = 1 - distance.chebyshev(hist1, hist2)

		return corr
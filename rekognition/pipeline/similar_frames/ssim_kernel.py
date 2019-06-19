from .similar_frames_kernel import SimilarFramesKernel
from skimage.measure import compare_ssim as ssim


class SSIM(SimilarFramesKernel):
	def __init__(self):
		super().__init__

	def compare(self, frame1, frame2):
		corr = 0

		corr = ssim(frame1, frame2, multichannel=True)

		return corr
import av
import cv2
from progress.bar import Bar
from rekognition.pipeline.data_handler import DataHandlerElem

class ImageHandlerElem(DataHandlerElem):

	def run(self, path_to_image):
		image = cv2.imread(path_to_image)

		# for frame in container.decode(stream):
		#     frames_rgb.append(frame.to_rgb().to_ndarray())

		#     bar.next()

		# bar.finish()

		# return frames_rgb
		return (image, 1)
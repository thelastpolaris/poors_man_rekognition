from .data_handler import DataHandlerElem
from progress.bar import Bar
import av
import cv2

class VideoFrames():
	def __init__(self, decoder):
		self._decoder = decoder
		self._counter = 0

	def get_frames(self, num_of_frames=1):
		# print("Extracting frames from video")

		frames_data = []
		frames_pts = []
		old_counter = self._counter

		for frame in self._decoder:
			image = frame.to_rgb().to_ndarray()

			img_height = image.shape[0]
			img_width = image.shape[1]

			# if img_width > 640:
			# 	size_ratio = 640/img_width
			# 	image = cv2.resize(image, dsize=(int(img_width*size_ratio), int(img_height*size_ratio)), interpolation=cv2.INTER_CUBIC)

			# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			frames_data.append(image)
			frames_pts.append(frame.pts)

			self._counter += 1
			if self._counter - num_of_frames >= old_counter:
				yield frames_data, frames_pts

		return None, None

class VideoHandlerElem(DataHandlerElem):
	def __init__(self, input_path, max_frames):
		self._num_of_images = 0
		self._current_frame = 0 #counter
		self.input_path = input_path
		self._max_frames = max_frames

	def run(self, data):
		container = av.open(self.input_path)
		# Get video stream
		stream = container.streams.video[0]

		data.frames_reader = VideoFrames(container.decode(stream))

	@property
	def max_frames(self):
		return self._max_frames

	@max_frames.setter
	def max_frames(self, max_frames):
		self._max_frames = max_frames
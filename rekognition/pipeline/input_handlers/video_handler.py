from .data_handler import DataHandlerElem
from progress.bar import Bar
import av
import cv2

class VideoFrames():
	def __init__(self, decoder, stream, max_frames = 0):
		self._decoder = decoder
		self._counter = 0
		self._stream = stream
		self._max_frames = max_frames

	@property
	def frames_num(self):
		return self._max_frames if self._max_frames else self._stream.frames

	def get_frames(self, num_of_frames=1):
		# print("Extracting frames from video")

		frames_data = []
		frames_pts = []
		old_counter = self._counter

		for frame in self._decoder:
			if self._max_frames > 0:
				if self._counter >= self._max_frames:
					return None, None

			image = frame.to_rgb().to_ndarray()

			#
			#
			# Add preprocessors here
			#
			#

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
				data, pts = frames_data, frames_pts
				frames_data, frames_pts = [], []
				yield data, pts

		return None, None

class VideoHandlerElem(DataHandlerElem):
	def __init__(self, input_path, max_frames = 0):
		self.input_path = input_path
		self._max_frames = max_frames

	def run(self, data):
		container = av.open(self.input_path)
		# Get video stream
		stream = container.streams.video[0]

		data.frames_reader = VideoFrames(container.decode(stream), stream, self._max_frames)

	@property
	def max_frames(self):
		return self._max_frames

	@max_frames.setter
	def max_frames(self, max_frames):
		self._max_frames = max_frames
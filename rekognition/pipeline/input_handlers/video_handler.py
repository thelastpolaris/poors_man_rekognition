from .data_handler import DataHandlerElem
import av

class VideoFrames():
	def __init__(self, container, stream, preprocessors = [], max_frames = 0, input_path = ""):
		self._container = container
		self._counter = 0
		self._stream = stream
		self._max_frames = max_frames
		self._preprocessors = preprocessors
		self.input_path = input_path

	@property
	def frames_num(self):
		return self._max_frames if self._max_frames else self._stream.frames

	def get_frames(self, num_of_frames = 1):
		self._container = av.open(self.input_path)

		decoder = self._container.decode(self._stream)
		self._counter = 0

		return self.frames_generator(decoder, num_of_frames)

	def frames_generator(self, decoder, num_of_frames):
		frames_data = []
		frames_pts = []
		old_counter = self._counter

		for frame in decoder:
			if self._max_frames > 0:
				if self._counter >= self._max_frames:
					return None, None

			image = frame.to_rgb().to_ndarray()

			# Preprocess the data
			if self._preprocessors:
				for p in self._preprocessors:
					image = p.process(image)

			self._counter += 1

			if num_of_frames == 1:
				yield image, frame.pts
			else:
				frames_data.append(image)
				frames_pts.append(frame.pts)

				if self._counter - num_of_frames >= old_counter:
					yield frames_data, frames_pts
					frames_data, frames_pts = [], []

		self._container.close()
		return None, None

class VideoHandlerElem(DataHandlerElem):
	def __init__(self, preprocessors = None):
		self.input_path = None
		self._max_frames = 0
		self._preprocessors = preprocessors

	def run(self, data, input_path, max_frames = 0):
		self.input_path = input_path
		self._max_frames = max_frames

		container = av.open(self.input_path)
		# Get video stream
		stream = container.streams.video[0]

		data.frames_reader = VideoFrames(container, stream, self._preprocessors, self._max_frames, self.input_path)

	@property
	def max_frames(self):
		return self._max_frames

	@max_frames.setter
	def max_frames(self, max_frames):
		self._max_frames = max_frames
import av
from ..pipeline_element import PipelineElement

class VideoFrames:
	def __init__(self, container, stream, preprocessors = [], max_frames = 0, input_path = ""):
		self._container = container
		self._counter = 0
		self._stream = stream
		self._max_frames = max_frames
		self._preprocessors = preprocessors
		self.input_path = input_path
		self._frames_group = None

	def frames_num(self, group_frames = True):
		if group_frames and self.frames_group:
			return len(self.frames_group)

		return self._max_frames if self._max_frames else self._stream.frames

	def get_frames(self, num_of_frames = 1, group_frames = True, first_frame=0, last_frame=0):
		self._container = av.open(self.input_path)

		decoder = self._container.decode(self._stream)
		self._counter = 0

		generator = self.frames_generator(decoder, num_of_frames, group_frames,
										  first_frame, last_frame)

		return generator

	@property
	def frames_group(self):
		return self._frames_group

	@frames_group.setter
	def frames_group(self, frames_group):
		self._frames_group = frames_group

	def frames_generator(self, decoder, num_of_frames, group_frames = True,
						 first_frame = 0, last_frame = 0):

		if first_frame and last_frame:
			if first_frame > last_frame or first_frame == last_frame:
				raise ValueError("Wrong interval provided to frames_generator")

		frames_data = []
		frames_pts = []
		old_counter = self._counter

		group_i = 0
		skip_frames = 0

		for i, frame in enumerate(decoder):
			if self._max_frames > 0:
				if i >= self._max_frames:
					return None, None

			if first_frame:
				if i < first_frame:
					continue

			if last_frame:
				if i >= last_frame:
					break

			if group_frames:
				if skip_frames:
					skip_frames -= 1
					continue

				if self.frames_group and not skip_frames:
					if group_i < len(self.frames_group):
						skip_frames = self.frames_group[group_i] - 1
						group_i += 1

			image = frame.to_rgb().to_ndarray()

			#Preprocess data
			if self._preprocessors:
				for p in self._preprocessors:
					image = p.process(image)

			if num_of_frames == 1:
				yield image, frame.pts

			else:
				frames_data.append(image)
				frames_pts.append(frame.pts)

				if i - num_of_frames >= old_counter:
					yield frames_data, frames_pts

					frames_data, frames_pts = [], []
					old_counter = i

		self._container.close()
		return None, None

class VideoHandlerElem(PipelineElement):
	def __init__(self, preprocessors = []):
		super().__init__()
		self.input_path = None
		self._max_frames = 0
		self._preprocessors = preprocessors

	def run(self, data, input_path, benchmark = False, max_frames = 0):
		self.input_path = input_path
		self._max_frames = max_frames

		container = av.open(self.input_path)
		# Get video stream
		stream = container.streams.video[0]

		data.add_value("frames_reader", VideoFrames(container, stream, self._preprocessors, self._max_frames, self.input_path))

	@property
	def max_frames(self):
		return self._max_frames

	@max_frames.setter
	def max_frames(self, max_frames):
		self._max_frames = max_frames

	def requires(self):
		return None

	def __str__(self):
		output = ""

		# Print preprocessors
		elems_len = len(self._preprocessors)

		for i in range(elems_len):
			elem = self._preprocessors[i]
			output += str(elem.__class__.__name__)

			if i != elems_len - 1:
				output += ", "

		return str(self.__class__.__name__) + "({})".format(output)

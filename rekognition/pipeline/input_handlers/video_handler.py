import av, cv2, abc
from .data_handler import DataHandlerElem, Data
from progress.bar import Bar


class Frame(Data):
	def __init__(self, pts, image_data):
		super().__init__(image_data)
		self._pts = pts

	def get_JSON(self):
		data = super().get_JSON()
		data["pts"] = self._pts

		return data


class VideoHandlerElem(DataHandlerElem):
	def run(self, input_data):
		container = av.open(self.input_path)
		# Get video stream
		stream = container.streams.video[0]
		self.num_of_images = stream.frames
		# stream.codec_context.skip_frame = 'NONKEY'

		bar = None

		print("Extracting frames from video")
		for frame in container.decode(stream):
			image = frame.to_rgb().to_ndarray()

			if bar is None:
				bar = Bar('Processing', max = self.num_of_images)
			
			img_height = image.shape[0]
			img_width = image.shape[1]

			if img_width > 640:
				size_ratio = 640/img_width
				image = cv2.resize(image, dsize=(int(img_width*size_ratio), int(img_height*size_ratio)), interpolation=cv2.INTER_CUBIC)

			self._current_frame += 1
			# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			if self._current_frame < self._max_frames:
				input_data.append(Frame(frame.pts, image))
				bar.next()
			else:
				bar.finish()
				break

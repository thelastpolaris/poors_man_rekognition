import av, cv2, abc
from .data_handler import DataHandlerElem, Data

class Frame(Data):
	def __init__(self, pts, image_data):
		super().__init__(image_data)
		self._pts = pts

	def get_JSON(self):
		data = super().get_JSON()
		data["pts"] = self._pts

		return data

class VideoHandlerElem(DataHandlerElem):

	def run(self, path_to_video):
		super().run(path_to_video)

		container = av.open(path_to_video)
		# Get video stream
		stream = container.streams.video[0]
		self.num_of_images = stream.frames
		# stream.codec_context.skip_frame = 'NONKEY'

		for frame in container.decode(stream):
			image = frame.to_rgb().to_ndarray()
			# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			yield Frame(frame.pts, image)

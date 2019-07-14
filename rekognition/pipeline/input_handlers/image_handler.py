from ..pipeline_element import PipelineElement
import os
import cv2
import glob

class ImagesReader:
	def __init__(self, input_path = "", preprocessors = None):
		self.input_path = input_path
		self._counter = 0
		self._preprocessors = preprocessors

	def frames_num(self):
		return len(glob.glob(os.path.join(self.input_path, "*")))

	def get_frames(self, num_of_frames=1):
		# select the path
		for file in glob.glob(os.path.join(self.input_path, "*")):
			image = cv2.imread(file)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			#Preprocess data
			if self._preprocessors:
				for p in self._preprocessors:
					image = p.process(image)

			if num_of_frames == 1:
				yield image, os.path.basename(file)

			# else:
			# 	frames_data.append(image)
			# 	frames_pts.append(frame.pts)
			#
			# 	if i - num_of_frames >= old_counter:
			# 		yield frames_data, frames_pts
			#
			# 		frames_data, frames_pts = [], []
			# 		old_counter = i

		return None, None

class ImageHandlerElem(PipelineElement):
	def __init__(self):
		super().__init__()
		self.input_path = None

	def run(self, data, input_path, benchmark = False, max_frames = 0, preprocessors = None):
		self.input_path = input_path

		data.add_value("frames_reader", ImagesReader(input_path, preprocessors))

	def requires(self):
		return None

	def get_JSON(self, data, json_objects):
		json_objects = [] # reset list with JSON values
		images_names = data.get_value("frames_pts")

		if images_names:
			for name in images_names:
				image = dict()
				image["filename"] = name
				json_objects.append(image)

		return json_objects
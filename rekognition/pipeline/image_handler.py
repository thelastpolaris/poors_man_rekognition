import os, cv2
from rekognition.pipeline.data_handler import DataHandlerElem, Data

class Image(Data):
	__name = None

	def __init__(self, image_data, name):
		super().__init__(image_data)
		__name = name

class ImageHandlerElem(DataHandlerElem):
	def run(self, path_to_folder):
		super().run(path_to_folder)
		
		if os.path.isdir(path_to_folder) == False:
			print("Folder not found")
			#raise fatal error
			return None
		
		for filename in os.listdir(path_to_folder):
			image = cv2.imread(path_to_folder + filename)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			yield Image(image, filename)



		# for frame in container.decode(stream):
		#     frames_rgb.append(frame.to_rgb().to_ndarray())

		#     bar.next()

		# bar.finish()

		# return frames_rgb
		# return (image, 1)
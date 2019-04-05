import os, cv2
from .data_handler import DataHandlerElem, Data

class Image(Data):
	def __init__(self, image_data, filename):
		super().__init__(image_data)
		self._filename = filename

	@property
	def filename(self):
		return self._filename

class ImageHandlerElem(DataHandlerElem):
	def run(self, path_to_folder):
		path_to_folder = path_to_folder + "/"
		super().run(path_to_folder)
		
		if os.path.isdir(path_to_folder) == False:
			print("Folder not found")
			#raise fatal error
			return None
		
		img_filenames = os.listdir(path_to_folder)
		self.num_of_images = len(img_filenames)

		for filename in img_filenames:
			image = cv2.imread(path_to_folder + filename)
			# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			yield Image(image, filename)
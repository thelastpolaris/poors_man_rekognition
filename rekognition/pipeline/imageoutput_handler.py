from rekognition.pipeline.output_handler import OutputHandler
from ..utils import visualization_utils_color as vis_util
from progress.bar import Bar
import cv2, os

class ImageOutputHandler(OutputHandler):
	def run(self, input_data):
		files_dir = self.parent_pipeline.filename
		
		print("Saving processed images")
		bar = Bar('Processing', max = len(input_data))
		
		new_files_dir = files_dir + "_output/"
		if not os.path.exists(new_files_dir):
			os.mkdir(new_files_dir)

		for data in input_data:
			image = data.image_data
			out_filename = os.path.splitext(data.filename)[0] + "_output.jpg"

			for face in data.faces:
				ymin, xmin, ymax, xmax = face.bounding_box
				name = face.person.predicted_name
				
				vis_util.draw_bounding_box_on_image_array(image,
												 ymin,
												 xmin,
												 ymax,
												 xmax,
												 display_str_list=[name])

			cv2.imwrite(new_files_dir + out_filename, image)
			bar.next()

		bar.finish()
		return input_data
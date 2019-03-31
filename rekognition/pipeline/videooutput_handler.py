from rekognition.pipeline.output_handler import OutputHandler
from ..utils import visualization_utils_color as vis_util
import cv2
import av

class VideoOutputHandler(OutputHandler):
	def run(self, input_data):
		out = None

		for data in input_data:
			image = data.get_image_data()

			self.get_parent_pipeline().get_filename()

			if out is None:
				[h, w] = image.shape[:2]
				out = cv2.VideoWriter("test_out.avi", 0, 25.0, (w, h))

			for face in data.get_faces():
				ymin, xmin, ymax, xmax = face.get_bounding_box()
				name = face.get_person().get_predicted_name()
				
				vis_util.draw_bounding_box_on_image_array(image,
												 ymin,
												 xmin,
												 ymax,
												 xmax,
												 display_str_list=[name] )

			out.write(image)

		out.release()

		return input_data
from .output_handler import OutputHandler
from ..input_handlers.image_handler import ImageHandlerElem
from ...utils import utils
from progress.bar import Bar
import cv2, os

class ImageOutputHandler(OutputHandler):
	def run(self, data, benchmark, output_name):

		if not os.path.exists("output"):
			os.mkdir("output")

		frames_reader = data.get_value("frames_reader")
		frames_generator = frames_reader.get_frames()

		print("Saving processed images")
		bar = Bar('Processing', max = frames_reader.frames_num())

		new_files_dir = os.path.join("output", output_name)
		print(new_files_dir)
		
		if not os.path.exists(new_files_dir):
			os.mkdir(new_files_dir)

		frames_face_boxes = data.get_value("frames_face_boxes")
		frames_face_names = data.get_value("frames_face_names")

		for i, (image_data, image_name) in enumerate(frames_generator):

			out_filename = os.path.splitext(image_name)[0] + "_output.jpg"

			face_boxes = frames_face_boxes[i] if frames_face_names else None
			names = frames_face_names[i] if frames_face_names else None

			image = utils.draw_faces(image_data, face_boxes, names)
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			cv2.imwrite(os.path.join(new_files_dir, out_filename), image)
			bar.next()

		bar.finish()

	def requires(self):
		return ImageHandlerElem
from rekognition.pipeline.output_handler import OutputHandler
from ..utils import visualization_utils_color as vis_util
import cv2
import av

class VideoOutputHandler(OutputHandler):
	def run(self, input_data):

		container = av.open(self.get_parent_pipeline().get_filename() + '.mp4', mode='w')
		stream = None

		fps = 25

		for data in input_data:
			image = data.get_image_data()

			if stream is None:
				[h, w] = image.shape[:2]
				stream = container.add_stream('mpeg4', rate=fps)
				stream.height = h
				stream.width = w
				stream.pix_fmt = 'yuv420p'

			# if out is None:
			# 	out = cv2.VideoWriter("test_out.avi", 0, 25.0, (w, h))

			for face in data.get_faces():
				ymin, xmin, ymax, xmax = face.get_bounding_box()
				name = face.get_person().get_predicted_name()
				
				vis_util.draw_bounding_box_on_image_array(image,
												 ymin,
												 xmin,
												 ymax,
												 xmax,
												 display_str_list=[name])

			frame = av.VideoFrame.from_ndarray(image, format='rgb24')
			for packet in stream.encode(frame):
				container.mux(packet)

        # flush stream
		for packet in stream.encode():
			container.mux(packet)
		
		container.close()

		return input_data
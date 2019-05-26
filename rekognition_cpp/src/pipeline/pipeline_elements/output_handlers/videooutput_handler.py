from .output_handler import OutputHandler
from ...utils import visualization_utils_color as vis_util
from progress.bar import Bar
import av, os

class VideoOutputHandler(OutputHandler):
	def run(self, input_data):
		initialized = False
		# bar = Bar('Processing', max = len(input_data))

		for data in input_data:
			image = data.image_data

			if initialized is not True:
				filename = os.path.splitext(self.parent_pipeline.filename)[0]

				if not os.path.exists("output"):
					os.mkdir("output")

				container = av.open("output/" + filename + '_output.ts', mode='w')
				stream = None

				fps = 25

				initialized = True

				# print("Saving processed video")

			if stream is None:
				[h, w] = image.shape[:2]
				print(h, w)
				stream = container.add_stream('h264', rate=fps)
				stream.height = h
				stream.width = w
				stream.pix_fmt = 'yuv420p'

			for face in data.faces:
				ymin, xmin, ymax, xmax = face.bounding_box
				name = face.person.predicted_name
				
				vis_util.draw_bounding_box_on_image_array(image,
												 ymin,
												 xmin,
												 ymax,
												 xmax,
												 display_str_list=[name])

			frame = av.VideoFrame.from_ndarray(image, format='rgb24')
			for packet in stream.encode(frame):
				container.mux(packet)
			# bar.next()

        # flush stream
		for packet in stream.encode():
			container.mux(packet)
		
		container.close()
		# bar.finish()

		return input_data
from .output_handler import OutputHandler
from ...utils import visualization_utils_color as vis_util
from ...utils import utils
from progress.bar import Bar
import av, os
from PIL import Image
import cv2

class VideoOutputHandler(OutputHandler):
	def run(self, data, output_name):
		if not os.path.exists("output"):
			os.mkdir("output")

		container = av.open("output/" + output_name + '_output.mp4', mode='w')
		stream = None

		fps = 25

		frames_reader = data.frames_reader

		print("Saving processed video")
		bar = Bar('Processing', max = frames_reader.frames_num(group_frames=False))

		frames_generator = frames_reader.get_frames(group_frames=False, return_key_frame = True)

		i = 0
		group_i = 0
		group = 0
		frames_group = data.frames_reader.frames_group

		for frames_data, frames_pts, frames_key in frames_generator:
			image = frames_data
			counter = i

			if stream is None:
				[h, w] = image.shape[:2]
				stream = container.add_stream('h264', rate=fps)
				stream.height = h
				stream.width = w

				if frames_group:
					if not group:
						group = frames_group[group_i]
						group_i += 1
					else:
						group -= 1

					counter = group_i + group

			if data._frames_face_boxes:
				frame_boxes = data._frames_face_boxes[counter]

				if len(frame_boxes):
					for f in range(len(frame_boxes)):
						ymin, xmin, ymax, xmax = frame_boxes[f]

						if data._frames_face_names:
							name = data._frames_face_names[counter][f][0]
						else:
							name = ""

						vis_util.draw_bounding_box_on_image_array(image,
														 ymin,
														 xmin,
														 ymax,
														 xmax,
														 display_str_list=[name],
														 use_normalized_coordinates = utils.is_normalized(frame_boxes[0]))

			# Key Frame
			color = 0 if frames_key else 125
			cv2.putText(image, "KEY", (int(w * 0.05), int(h * 0.05)), cv2.FONT_HERSHEY_DUPLEX, 1, color)

			if data._frames_correlation:
				color = 0
				cor = data._frames_correlation[i]
				if cor < 0.97:
					color = 255

				cv2.putText(image, str(cor), (int(w*0.05), int(h*0.95)), cv2.FONT_HERSHEY_DUPLEX, 1, color)

			frame = av.VideoFrame.from_ndarray(image, format='rgb24')
			for packet in stream.encode(frame):
				container.mux(packet)

			i += 1
			bar.next()

		# flush stream
		for packet in stream.encode():
			container.mux(packet)
		
		container.close()
		bar.finish()
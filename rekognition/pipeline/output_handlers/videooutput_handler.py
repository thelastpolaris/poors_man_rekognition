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

		frames_generator = frames_reader.get_frames(group_frames=False)

		frames_group = data.frames_reader.frames_group

		enum_frames = enumerate(frames_generator)
		if frames_group:
			group_i = 0
			group = frames_group[group_i] + 1

		for i, (frames_data, frames_pts) in enum_frames:
			image = frames_data
			counter = i

			if stream is None:
				[h, w] = image.shape[:2]
				stream = container.add_stream('h264', rate=fps)
				stream.height = h
				stream.width = w

			if frames_group:
				if i < group:
					counter = group_i
				else:
					group_i += 1
					new_group = frames_group[group_i]
					if not new_group:
						new_group = 1
					group += new_group
					counter = group_i

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

			if data._frames_correlation:
				color = 0
				cor = data._frames_correlation[i]
				if cor < 0.97:
					color = 255

				cv2.putText(image, str(cor), (int(w*0.05), int(h*0.95)), cv2.FONT_HERSHEY_DUPLEX, 1, color)

			frame = av.VideoFrame.from_ndarray(image, format='rgb24')
			for packet in stream.encode(frame):
				container.mux(packet)

			bar.next()

		# flush stream
		for packet in stream.encode():
			container.mux(packet)
		
		container.close()
		bar.finish()
from .output_handler import OutputHandler
from ...utils import utils
from progress.bar import Bar
import av, os
from ..input_handlers.video_handler import VideoHandlerElem

class VideoOutputHandler(OutputHandler):
	def run(self, data, benchmark, output_name):
		if not os.path.exists("output"):
			os.mkdir("output")

		container = av.open("output/" + output_name + '_output.mp4', mode='w')
		stream = None

		fps = 25

		frames_reader = data.get_value("frames_reader")

		print("Saving processed video")
		bar = Bar('Processing', max = frames_reader.frames_num(group_frames=False))

		frames_generator = frames_reader.get_frames(group_frames=False)
		frames_group = frames_reader.frames_group

		frames_face_boxes = data.get_value("frames_face_boxes")
		frames_face_names = data.get_value("frames_face_names")

		if frames_group:
			group_i = 0
			group = frames_group[group_i] + 1

		for i, (frames_data, frames_pts) in enumerate(frames_generator):
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

				face_boxes = frames_face_boxes[counter] if frames_face_names else None
				names = frames_face_names[counter] if frames_face_names else None

				image = utils.draw_faces(image, face_boxes, names)

			frame = av.VideoFrame.from_ndarray(image, format='rgb24')
			for packet in stream.encode(frame):
				container.mux(packet)

			bar.next()

		# flush stream
		for packet in stream.encode():
			container.mux(packet)
		
		container.close()
		bar.finish()

	def requires(self):
		return VideoHandlerElem
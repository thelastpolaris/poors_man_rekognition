import av
from progress.bar import Bar
from rekognition.pipeline.pipeline_element import PipelineElement

class HandleVideoElem(PipelineElement):

	def run(self, path_to_video):
		container = av.open(path_to_video)

		# Get video stream
		stream = container.streams.video[0]
		# stream.codec_context.skip_frame = 'NONKEY'

		frames_rgb = []

		print("Starting extracting frames from video")
		bar = Bar('Processing', max=stream.frames)

		for frame in container.decode(stream):
		    frames_rgb.append(frame.to_rgb().to_ndarray())

		    bar.next()

		bar.finish()

		return frames_rgb
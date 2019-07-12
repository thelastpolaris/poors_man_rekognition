from ..pipeline_element import PipelineElement
from ..input_handlers.video_handler import VideoHandlerElem

class SimilarFramesFinder(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark=False, sim_threshold = 0.97, max_jobs = 4, max_group = 30):
		frames_reader = data.get_value("frames_reader")
		frames_pts, frames_correlation, benchmark_data = self.kernel.run(frames_reader, benchmark, max_jobs)

		data.add_value("frames_pts", frames_pts)
		data.add_value("frames_correlation", frames_correlation)

		sim_count = 0
		frames_group = []

		for i, corr in enumerate(frames_correlation):
			sim_count += 1
			if corr < sim_threshold:
				frames_group.append(sim_count)
				sim_count = 0

		if sim_count:
			frames_group.append(sim_count)

		data.add_value("frames_group", frames_group)
		frames_reader.frames_group = frames_group

		benchmark_data["Group Frames"] = len(frames_group)

		if benchmark:
			self.benchmark(data, benchmark_data)

	def requires(self):
		return VideoHandlerElem

	def benchmark(self, data, benchmark_data):
		for k, v in benchmark_data.items():
			if k != "scores":
				data.benchmark.add_value(self, k, v)

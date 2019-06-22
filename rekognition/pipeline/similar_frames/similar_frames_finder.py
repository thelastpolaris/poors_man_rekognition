from ..pipeline_element import PipelineElement

class SimilarFramesFinder(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark=False, sim_threshold = 0.97, max_jobs = 4, max_group = 30):
		data._frames_pts, data._frames_correlation, benchmark_data = self.kernel.run(data.frames_reader, benchmark, max_jobs)

		sim_count = 0
		frames_group = []


		for i, corr in enumerate(data._frames_correlation):
			sim_count += 1
			if corr < sim_threshold:
				frames_group.append(sim_count)
				sim_count = 0

		if sim_count:
			frames_group.append(sim_count)

		data.frames_reader.frames_group = frames_group

		benchmark_data["Similar Frames"] = len(frames_group)

		if benchmark:
			self.benchmark(data, benchmark_data)

	def benchmark(self, data, benchmark_data):
		for k, v in benchmark_data.items():
			if k != "scores":
				data.benchmark.add_value(self, k, v)

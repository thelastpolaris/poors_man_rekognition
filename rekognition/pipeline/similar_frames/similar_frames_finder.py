from ..pipeline_element import PipelineElement

class SimilarFramesFinder(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark=False, sim_threshold = 0.97):
		data._frames_pts, data._frames_correlation, benchmark_data = self.kernel.run(data.frames_reader, benchmark)

		sim_count = 0
		frames_group = []

		for i, corr in enumerate(data._frames_correlation):
			if corr > sim_threshold:
				sim_count += 1
			else:
				frames_group.append(sim_count)
				sim_count = 0

		if sim_count:
			frames_group.append(sim_count)

		data.frames_reader.frames_group = frames_group
from ..pipeline_element import PipelineElement
from ..input_handlers.video_handler import VideoHandlerElem

import pickle
import os

class SimilarFramesFinder(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark=False, sim_threshold = 0.97, max_jobs = 4, max_group = 30, serialize_dir = ""):
		benchmark_data = {}
		frames_reader = data.get_value("frames_reader")

		frames_group_file = os.path.join(serialize_dir, "frames_group.pkl")
		frames_pts_file = os.path.join(serialize_dir, "frames_pts.pkl")

		if serialize_dir and os.path.isfile(frames_group_file) and os.path.isfile(frames_pts_file):
			with open(frames_group_file, 'rb') as f:
				frames_group = pickle.load(f)

			with open(os.path.join(serialize_dir, frames_pts_file), 'rb') as f:
				frames_pts = pickle.load(f)
		else:
			frames_pts, frames_correlation, benchmark_data = self.kernel.run(frames_reader, benchmark, max_jobs)
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

			if serialize_dir:
				if not os.path.isfile(frames_group_file):
					with open(frames_group_file, 'wb') as f:
						pickle.dump(frames_group, f)

				if not os.path.isfile(frames_pts_file):
					with open(frames_pts_file, 'wb') as f:
						pickle.dump(frames_pts, f)

		data.add_value("frames_pts", frames_pts)
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

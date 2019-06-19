from ..kernel import Kernel
import time
from progress.bar import Bar
import abc
import numpy as np
import multiprocessing

class SimilarFramesKernel(Kernel):
	def __init__(self):
		super().__init__(run_as_process=False)

	@abc.abstractmethod
	def compare(self, frame1, frame2):
		pass

	def predict(self, connection, frames_reader, benchmark: bool, max_jobs, batch_size = 0):
		benchmark_data = None
		if benchmark:
			benchmark_data = dict()

		print("Detecting similar frames in video")

		if benchmark:
			start = time.time()

		all_frames_num = frames_reader.frames_num()
		frames_per_job = int(all_frames_num/max_jobs)
		processes = []
		connections = []

		for job_id in range(max_jobs):
			first_frame = job_id*frames_per_job
			last_frame = 0

			if job_id != max_jobs - 1:
				last_frame = first_frame + frames_per_job

			frames_generator = frames_reader.get_frames(first_frame=first_frame, last_frame=last_frame)

			def process_frames(connection, frames_generator, target_func, job_id, frames_num, verbose = 50):
				prev_frame = np.array([])
				sub_frames_correlation = []
				sub_frames_pts = []

				print("Started job # {}".format(job_id))

				i = 0

				for frames_data, frames_pts in frames_generator:
					corr = 0

					if not prev_frame.any():
						prev_frame = frames_data
					else:
						corr = target_func(prev_frame, frames_data)
						prev_frame = frames_data

					sub_frames_correlation.append(corr)
					sub_frames_pts.append(frames_pts)
					i += 1
					if not i % verbose:
						print("Job # {} - {}/{}".format(job_id, i, frames_num))

				connection.send((job_id, sub_frames_correlation, sub_frames_pts))

			parent_conn, child_conn = multiprocessing.Pipe()
			connections.append((parent_conn, child_conn))

			frames_num = abs(last_frame - first_frame)
			if not last_frame:
				frames_num = all_frames_num - first_frame

			process = multiprocessing.Process(target=process_frames,
											  args=(parent_conn, frames_generator, self.compare, job_id, frames_num))
			process.daemon = True

			processes.append(process)

		for p in processes:
			p.start()

		all_frames_pts = []
		all_frames_correlation = []

		for c in connections:
			results = c[1].recv()

			all_frames_correlation += results[1]
			all_frames_pts += results[2]

		for p in processes:
			p.join()

		if benchmark:
			end = time.time()
			benchmark_data["Grouping Time"] = end - start

		connection.send((all_frames_pts, all_frames_correlation, benchmark_data))
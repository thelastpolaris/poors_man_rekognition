import av
from progress.bar import Bar
from ..pipeline_element import PipelineElement
import multiprocessing

class FaceDetectorElem(PipelineElement):
	def __init__(self, kernel):
		self._kernel = kernel

	def run(self, data):
		parent_conn, child_conn = multiprocessing.Pipe()
		p1 = multiprocessing.Process(target=self._kernel.run, args=(data.frames_reader, parent_conn, ))

		p1.start()
		frames_faces, frames_pts = child_conn.recv()
		p1.join()

		print(frames_pts)
		data._frames_faces = frames_faces
		data._frames_pts = frames_pts

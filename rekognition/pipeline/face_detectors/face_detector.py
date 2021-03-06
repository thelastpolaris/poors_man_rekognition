from ..pipeline_element import PipelineElement
from ...utils import utils

from ..input_handlers.video_handler import VideoHandlerElem
from ..input_handlers.video_handler import VideoFrames

from ..input_handlers.image_handler import ImageHandlerElem

import pickle
import os

class FaceDetectorElem(PipelineElement):
	def __init__(self, kernel):
		super().__init__(kernel)

	def run(self, data, benchmark=False, benchmark_boxes = None, min_score = 0.7, face_tracking = True, serialize_dir = ""):
		benchmark_data = {}
		frames_reader = data.get_value("frames_reader")

		face_boxes_file = os.path.join(serialize_dir, "frames_face_boxes.pkl")
		frames_pts_file = os.path.join(serialize_dir, "frames_pts.pkl")

		if serialize_dir and os.path.isfile(face_boxes_file) and os.path.isfile(frames_pts_file):
			with open(face_boxes_file, 'rb') as f:
				frames_face_boxes = pickle.load(f)

			with open(frames_pts_file, 'rb') as f:
				frames_pts = pickle.load(f)
		else:
			frames_face_boxes, frames_pts, benchmark_data = self.kernel.run(frames_reader, benchmark, min_score)

			if serialize_dir:
				if not os.path.isfile(face_boxes_file):
					with open(face_boxes_file, 'wb') as f:
						pickle.dump(frames_face_boxes, f)

				if not os.path.isfile(frames_pts_file):
					with open(frames_pts_file, 'wb') as f:
						pickle.dump(frames_pts, f)

		data.add_value("frames_face_boxes", frames_face_boxes)
		data.add_value("frames_pts", frames_pts)

		if face_tracking and type(frames_reader) == VideoFrames:
			tracked_faces = self.face_tracking(frames_face_boxes)

			data.add_value("tracked_faces", tracked_faces)

		if benchmark:
			self.benchmark(data, benchmark_data, benchmark_boxes)

	@staticmethod
	def face_tracking(frames_face_boxes):
			persons = [[(0, i)] for i in range(len(frames_face_boxes[0]))]
			persons_frames = [[i, face_box, False] for i, face_box in enumerate(frames_face_boxes[0])]

			for i, face_boxes in enumerate(frames_face_boxes):
				if i == 0:
					continue

				for b, box in enumerate(face_boxes):
					found = False
					for p, person_f in enumerate(persons_frames):
						if utils.IoU(person_f[1], box) > utils.IOU_THRESHOLD:
							persons[person_f[0]].append((i, b))
							persons_frames[p][1] = box
							persons_frames[p][2] = True
							found = True
							break

					if not found:
						persons.append([(i, b)])
						persons_frames.append([len(persons) - 1, box, True])

				for p, person_f in enumerate(persons_frames):
					if person_f[2]:
						persons_frames[p][2] = False
					else:
						del persons_frames[p]

			return persons

	def requires(self):
		return VideoHandlerElem, ImageHandlerElem

	def get_JSON(self, data, json_holder):
		frames_group = data.get_value("frames_group")
		frames_face_boxes = data.get_value("frames_face_boxes")
		json_objects = json_holder["frames"]

		for (i, all_count) in utils.traverse_group(len(frames_face_boxes), frames_group):
			faces = []

			for bb in frames_face_boxes[i]:
				face = dict()
				face["bounding_box"] = {"top": float(bb[0]), "left": float(bb[1]), "bottom": float(bb[2]),
										"right": float(bb[3])}
				faces.append(face)

			json_objects[all_count]["faces"] = faces

		json_holder["frames"] = json_objects
		return json_holder

	def benchmark(self, data, benchmark_data, benchmark_boxes):
		for k, v in benchmark_data.items():
			if k != "scores":
				data.benchmark.add_value(self, k, v)

		if "scores" in benchmark_data.keys():
			scores = benchmark_data["scores"]

		if benchmark_boxes != None:
			bench_boxes, bench_w, bench_h, bench_labels = utils.boxes_from_cvat_xml(benchmark_boxes)

			if bench_boxes:
				TP = 0
				FP = 0
				FN = 0

				frames_group = data.get_value("frames_reader").frames_group

				bench_count = 0

				for (i, frame_boxes) in enumerate(data.get_value("frames_face_boxes")):
					group = 1
					if frames_group:
						group = frames_group[i]

					for a in range(group):
						bench_count += 1
						if bench_count < len(bench_boxes) - 1:
							_TP, _FP, _FN = utils.calculate_tp_fp_fn(frame_boxes, bench_boxes[bench_count], bench_w, bench_h)
							TP += _TP
							FP += _FP
							FN += _FN

				precision = 0 if (TP+FP) == 0 else TP/(TP + FP)
				recall = 0 if (TP + FN) == 0 else TP/(TP + FN)

				data.benchmark.add_value(self, "Precision", precision)
				data.benchmark.add_value(self, "Recall", recall)
import numpy as np
import os
from progress.bar import Bar
from ..kernel import Kernel

import torch
from ...model.dsfd.face_ssd_infer import SSD

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class DSFDFaceDetector(Kernel):
	def __init__(self, min_score=.5):
		super().__init__()
		self._model_path = parentDir + "/../model/dsfd/dsfd.pth"
		self._min_score = min_score


	def predict(self, connection, frames_reader):
		device = torch.device("cuda")

		net = SSD("Inference")
		net.load_state_dict(torch.load(self._model_path))
		net.to(device).eval()

		print("Detecting faces in video")
		bar = None
		i = 0

		all_frames_pts = []
		all_frames_face_boxes = []

		frames_generator = frames_reader.get_frames(1)

		for frames_data, frames_pts in frames_generator:
			frame_boxes = []

			if bar is None:
				bar = Bar('Processing', max = frames_reader.frames_num)

			target_size = (600, 600)
			print(target_size)
			frame_boxes = net.detect_on_image(frames_data, target_size, device, is_pad=False, keep_thresh=self._min_score)

			print(frame_boxes)

			all_frames_face_boxes.append(frame_boxes)
			all_frames_pts.append(frames_pts)

			i += 1
			bar.next()

		if bar:
			bar.finish()

		connection.send((all_frames_face_boxes, all_frames_pts))

		return
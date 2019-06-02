import numpy as np
import sys, os
import tensorflow as tf
from progress.bar import Bar
from ..kernel import Kernel

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

from ...utils import label_map_util
from ...utils import visualization_utils_color as vis_util

# from .face_detector import FaceDetectorElem

class MobileNetsSSDFaceDetector(Kernel):
	def __init__(self, min_score=.7):
		super().__init__()
		self._model_path = parentDir + '/../model/mnssd_frozen_graph.pb'
		self._labels_path = parentDir + '/../protos/face_label_map.pbtxt'

		self._min_score = min_score

		self._classes_num = 2

		self._label_map = None
		self._categories = None
		self._category_index = None

		self._detection_graph = None
		self._config = None

		self._label_map = label_map_util.load_labelmap(self._labels_path)
		self._categories = label_map_util.convert_label_map_to_categories(self._label_map, max_num_classes=self._classes_num, use_display_name=True)
		self._category_index = label_map_util.create_category_index(self._categories)

		self._detection_graph = tf.Graph()
		with self._detection_graph.as_default():
			od_graph_def = tf.GraphDef()

			with tf.gfile.GFile(self._model_path, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		with self._detection_graph.as_default():
			self._config = tf.ConfigProto()
			# self._config.gpu_options.allow_growth = True
			# self._config.gpu_options.per_process_gpu_memory_fraction = 0.5

	def predict(self, connection, frames_reader):
		sess = tf.Session(graph=self._detection_graph, config=self._config)

		print("Detecting faces in video")
		bar = None
		i = 0

		all_frames_pts = []
		all_frames_face_boxes = []

		frames_generator = frames_reader.get_frames()

		for frames_data, frames_pts in frames_generator:
			i += 1
			image = frames_data

			if bar is None:
				bar = Bar('Processing', max = frames_reader.frames_num)

			image_expanded = np.expand_dims(image, axis=0)
			image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')

			boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')

			scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
			classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')

			(boxes, scores, classes, num_detections) = sess.run(
				[boxes, scores, classes, num_detections],
				feed_dict={image_tensor: image_expanded})

			bar.next()

			frame_boxes = []
			scores = np.squeeze(scores)
			boxes = np.squeeze(boxes)

			for b in range(len(boxes)):
				if scores[b] > self._min_score:
					frame_boxes.append(boxes[b])

			all_frames_face_boxes.append(frame_boxes)
			all_frames_pts.append(frames_pts)

		if bar:
			bar.finish()

		connection.send((all_frames_face_boxes, all_frames_pts))

		return
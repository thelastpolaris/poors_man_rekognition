import numpy as np
import sys, os
from .face_detector_kernel import FaceDetectorKernel
from ...utils import label_map_util

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)


class MobileNetsSSDFaceDetector(FaceDetectorKernel):
	def __init__(self):
		super().__init__()
		self._model_path = parentDir + '/../model/mnssd_frozen_graph.pb'
		self._labels_path = parentDir + '/../protos/face_label_map.pbtxt'

		self._classes_num = 2

		self._label_map = None
		self._categories = None
		self._category_index = None

		self._detection_graph = None
		self._config = None

		self._label_map = label_map_util.load_labelmap(self._labels_path)
		self._categories = label_map_util.convert_label_map_to_categories(self._label_map, max_num_classes=self._classes_num, use_display_name=True)
		self._category_index = label_map_util.create_category_index(self._categories)
			# self._config.gpu_options.allow_growth = True
			# self._config.gpu_options.per_process_gpu_memory_fraction = 0.5

	def load_model(self):
		import tensorflow as tf

		self._detection_graph = tf.Graph()
		self._sess = None

		with self._detection_graph.as_default():
			od_graph_def = tf.GraphDef()

			with tf.gfile.GFile(self._model_path, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		with self._detection_graph.as_default():
			self._config = tf.ConfigProto()

		self._sess = tf.Session(graph=self._detection_graph, config=self._config)

	def inference(self, image):
		image_expanded = np.expand_dims(image, axis=0)
		image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')

		boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')

		scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
		classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')

		(boxes, scores, classes, num_detections) = self._sess.run(
			[boxes, scores, classes, num_detections],
			feed_dict={image_tensor: image_expanded})

		scores = np.squeeze(scores)
		boxes = np.squeeze(boxes)

		return scores, boxes

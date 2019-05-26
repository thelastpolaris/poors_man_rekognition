# Significant portion of the code is adopted from the following package that is licensed by Apache License 2.0
# https://github.com/yeephycho/tensorflow-face-detection

import numpy as np
import tensorflow as tf
import sys, os
from progress.bar import Bar

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

from ...utils import label_map_util
from ...utils import visualization_utils_color as vis_util

from .face_detector import FaceDetectorElem
import sys
import objgraph

class MobileNetsSSDFaceDetector(FaceDetectorElem):
	def __init__(self, min_score_thresh=.7):
		super().__init__()
		self._model_path = parentDir + '/../model/mnssd_frozen_graph.pb'
		self._labels_path = parentDir + '/../protos/face_label_map.pbtxt'

		self._min_score_thresh = min_score_thresh

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
			self._config.gpu_options.allow_growth = True

	def run(self, input_data):
		# tf.reset_default_graph()
		sess = tf.Session(graph=self._detection_graph, config=self._config)

		faces = []
		frames = []

		print("Detecting faces in video")
		bar = None
		i = 0
		
		for data in input_data:
			i += 1

			image = data.image_data

			if bar is None:
				bar = Bar('Processing', max = self.parent_pipeline.num_of_images)

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

			frame_faces, face_boxes = vis_util.get_image_from_bounding_box(
				image,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				self._category_index,
				use_normalized_coordinates=True,
				min_score_thresh=self._min_score_thresh)

			for f in range(len(frame_faces)):
				data.add_face(frame_faces[f], face_boxes[f])
				# vis_util.save_image_array_as_png(frame_faces[f], "images/{}_{}.png".format(i, f))

			# print("refcount {}".format(sys.getrefcount(image)))

			# frames.append(data)
			yield data
			# if i > 100:
				# break

		# bar.finish()

		# return frames
		
import numpy as np
import tensorflow as tf
import sys, os
import time
import cv2
from progress.bar import Bar

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

from ..utils import label_map_util
from ..utils import visualization_utils_color as vis_util

from .pipeline_element import PipelineElement

class MobileNetsSSDFaceDetector(PipelineElement):
	__model_path = parentDir + '/model/frozen_inference_graph_face.pb'
	__labels_path = parentDir + '/protos/face_label_map.pbtxt'

	__classes_num = 2

	__label_map = None
	__categories = None
	__category_index = None

	__detection_graph = None
	__config = None

	def __init__(self):
		super().__init__()
		self.__label_map = label_map_util.load_labelmap(self.__labels_path)
		self.__categories = label_map_util.convert_label_map_to_categories(self.__label_map, max_num_classes=self.__classes_num, use_display_name=True)
		self.__category_index = label_map_util.create_category_index(self.__categories)

		self.__detection_graph = tf.Graph()
		with self.__detection_graph.as_default():
			od_graph_def = tf.GraphDef()

			with tf.gfile.GFile(self.__model_path, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		with self.__detection_graph.as_default():
			self.__config = tf.ConfigProto()
			self.__config.gpu_options.allow_growth = True

	def run(self, input_data):
		sess = tf.Session(graph=self.__detection_graph, config=self.__config)

		print("Detecting faces in video")
		bar = Bar('Processing', max = len(input_data))
		out = None

		for image in input_data:
			if out is None:
				[h, w] = image.shape[:2]
				out = cv2.VideoWriter("test_out.avi", 0, 25.0, (w, h))

			image_expanded = np.expand_dims(image, axis=0)
			image_tensor = self.__detection_graph.get_tensor_by_name('image_tensor:0')

			boxes = self.__detection_graph.get_tensor_by_name('detection_boxes:0')

			scores = self.__detection_graph.get_tensor_by_name('detection_scores:0')
			classes = self.__detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = self.__detection_graph.get_tensor_by_name('num_detections:0')

			start_time = time.time()
			(boxes, scores, classes, num_detections) = sess.run(
				[boxes, scores, classes, num_detections],
				feed_dict={image_tensor: image_expanded})
			
			elapsed_time = time.time() - start_time
			bar.next()
			# print('inference time cost: {}'.format(elapsed_time))

			vis_util.visualize_boxes_and_labels_on_image_array(
				image,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				self.__category_index,
				use_normalized_coordinates=True,
				line_thickness=4)

			out.write(image)

		bar.finish()
		out.release()
		
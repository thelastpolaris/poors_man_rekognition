import numpy as np
import tensorflow as tf
import sys, os
import time
import cv2
import av
from progress.bar import Bar

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

from ..utils import label_map_util
from ..utils import visualization_utils_color as vis_util

from .face_detector import FaceDetectorElem

class MobileNetsSSDFaceDetector(FaceDetectorElem):
	__model_path = parentDir + '/model/mnssd_frozen_graph.pb'
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
		tf.reset_default_graph()
		sess = tf.Session(graph=self.__detection_graph, config=self.__config)

		faces = []
		frames = []

		print("Detecting faces in video")
		bar = None
		i = 0
		
		for data in input_data:
			i += 1
			image = data.get_image_data()

			if bar is None:
				bar = Bar('Processing', max = self.parent_pipeline.num_of_images)

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

			frame_faces, face_boxes = vis_util.get_image_from_bounding_box(
				image,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				self.__category_index,
				use_normalized_coordinates=True,
				min_score_thresh=.6)

			for f in range(len(frame_faces)):
				data.add_face(frame_faces[f], face_boxes[f])
				# vis_util.save_image_array_as_png(frame_faces[f], "images/{}_{}.png".format(i, f))

			frames.append(data)

			# if i > 2:
				# break

		bar.finish()

		return frames
		
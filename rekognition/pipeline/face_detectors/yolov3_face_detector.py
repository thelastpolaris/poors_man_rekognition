import numpy as np
import os
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from PIL import Image

from ...model.yolov3.model import eval
from .face_detector_kernel import FaceDetectorKernel

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class YOLOv3FaceDetector(FaceDetectorKernel):
	def __init__(self, min_score=.5, iou = 0.45):
		super().__init__()
		self._model_path = parentDir + "/../model/yolov3/YOLO_Face.h5"
		self._class_names = ["face"]

		self._anchors = np.array([10.0, 13.0, 16.0, 30.0, 33.0, 23.0,
								  30.0, 61.0, 62.0, 45.0, 59.0, 119.0,
								  116.0, 90.0, 156.0, 198.0, 373.0, 326.0]).reshape(-1,2)

		self._min_score = min_score
		self._iou = iou

	def _generate(self):
		model_path = os.path.expanduser(self._model_path)
		assert model_path.endswith(
			'.h5'), 'Keras model or weights must be a .h5 file'

		# load model, or construct model and load weights
		num_anchors = len(self._anchors)
		num_classes = len(self._class_names)
		try:
			self.yolo_model = load_model(model_path, compile=False)
		except:
			# make sure model, anchors and classes match
			self.yolo_model.load_weights(self._model_path)
		else:
			assert self.yolo_model.layers[-1].output_shape[-1] == \
				   num_anchors / len(self.yolo_model.output) * (
						   num_classes + 5), \
				'Mismatch between model and given anchor and class sizes'
		print(
			'*** {} model, anchors, and classes loaded.'.format(model_path))

		# generate output tensor targets for filtered bounding boxes.
		self.input_image_shape = K.placeholder(shape=(2,))
		boxes, scores, classes = eval(self.yolo_model.output, self._anchors,
									  len(self._class_names),
									  self.input_image_shape,
									  score_threshold=self._min_score,
									  iou_threshold=self._iou)
		return boxes, scores, classes

	def letterbox_image(self, image, size):
		'''Resize image with unchanged aspect ratio using padding'''

		img_width, img_height = image.size
		w, h = size
		scale = min(w / img_width, h / img_height)
		nw = int(img_width * scale)
		nh = int(img_height * scale)

		image = image.resize((nw, nh), Image.BICUBIC)
		new_image = Image.new('RGB', size, (128, 128, 128))
		new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
		return new_image

	def load_model(self):
		self._sess = K.get_session()
		self._boxes, self._scores, self._classes = self._generate()

	def inference(self, image):
		image = Image.fromarray(image)

		new_image_size = (image.width - (image.width % 32),
						  image.height - (image.height % 32))
		boxed_image = self.letterbox_image(image, new_image_size)

		image_data = np.array(boxed_image, dtype='float32')

		image_data /= 255.
		# add batch dimension
		image_data = np.expand_dims(image_data, 0)
		boxes, scores, classes = self._sess.run(
			[self._boxes, self._scores, self._classes],
			feed_dict={
				self.yolo_model.input: image_data,
				self.input_image_shape: [image.size[1], image.size[0]],
				K.learning_phase(): 0
			})

		return scores, boxes
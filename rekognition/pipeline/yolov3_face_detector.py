import numpy as np
import cv2, os
from progress.bar import Bar
from .face_detector import FaceDetectorElem

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom

class YOLOv3FaceDetector(FaceDetectorElem):
	def __init__(self):
		super().__init__()
		model_weights = parentDir + "/model/yolov3/yolov3-wider_16000.weights"
		model_cfg = parentDir + "/model/yolov3/yolov3-face.cfg"

		self._net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
		self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
		self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

	# Get the names of the output layers
	def get_outputs_names(self):
		# Get the names of all the layers in the network
		layers_names = self._net.getLayerNames()

		# Get the names of the output layers, i.e. the layers with unconnected
		# outputs
		return [layers_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]


	def get_boxes(self, outs, conf_threshold, nms_threshold, image):
		image_height = image.shape[0]
		image_width = image.shape[1]

		# Scan through all the bounding boxes output from the network and keep only
		# the ones with high confidence scores. Assign the box's class label as the
		# class with the highest score.
		confidences = []
		boxes = []
		final_boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > conf_threshold:
					center_x = int(detection[0] * image_width)
					center_y = int(detection[1] * image_height)
					width = int(detection[2] * image_width)
					height = int(detection[3] * image_height)
					left = int(center_x - width / 2)
					top = int(center_y - height / 2)
					confidences.append(float(confidence))
					boxes.append([left, top, width, height])

		# Perform non maximum suppression to eliminate redundant
		# overlapping boxes with lower confidences.
		indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

		for i in indices:
			i = i[0]
			box = boxes[i]
			left = box[0]
			top = box[1]
			width = box[2]
			height = box[3]
			final_boxes.append(boxes[i])

		return final_boxes 


	def run(self, input_data):
		frames = []
		bar = None
		i = 0

		for data in input_data:
			i += 1
			if bar is None:
				bar = Bar('Processing', max = self.parent_pipeline.num_of_images)

			# Create a 4D blob from a frame.
			image = data.image_data
			
			image_height = image.shape[0]
			image_width = image.shape[1]

			blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)

        	# Sets the input to the network
			self._net.setInput(blob)

        	# Runs the forward pass to get output of the output layers
			outs = self._net.forward(self.get_outputs_names())
			
			boxes = self.get_boxes(outs, 0.5, 0.4, image)
			for box in boxes:
				left, top, right, bottom = refined_box(box[0], box[1], box[2], box[3])

				face = image[int(top):int(bottom), int(left):int(right)]
				# Output relative coordinates
				data.add_face(face, [top/image_height, left/image_width, bottom/image_height, right/image_width])

			frames.append(data)

			bar.next()

			# if i > 1:
				# break
		
		bar.finish()
		return frames
import numpy as np
import xmltodict
from . import visualization_utils_color as vis_util

IOU_THRESHOLD = 0.5

def extract_boxes(image, boxes):
	height = image.shape[0]
	width = image.shape[1]

	box_images = []

	for box in boxes:
		for i in range(len(box)):
			if box[i] < 0:
				box[i] = 0

			if box[i] > 1:
				box[i] = 1.0

		# Normalized coordinates
		if is_normalized(box):
			p1 = (int(box[0] * height), int(box[1] * width))
			p2 = (int(box[2] * height), int(box[3] * width))
		else:
			p1 = (int(box[0]), int(box[1]))
			p2 = (int(box[2]), int(box[3]))

		box_images.append(image[p1[0]:p2[0], p1[1]:p2[1]])

	return box_images

def is_normalized(box):
	if (box <= 1).all():
		return True
	return False

def boxes_from_cvat_xml(path_to_xml):
	with open(path_to_xml) as fd:
		doc = xmltodict.parse(fd.read())
		root = doc["annotations"]

		task = root["meta"]["task"]
		frames_num = int(task["size"])
		resolution = task["original_size"]
		width, height = int(resolution["width"]), int(resolution["height"])

		frames_boxes = [[] for i in range(frames_num)]
		frames_labels = [[] for i in range(frames_num)]

		# Extract boxes
		for track in root["track"]:
			label = track["@label"]

			for box in track["box"]:
				frame = int(box["@frame"])
				box_array = np.array([int(float(box["@ytl"])), int(float(box["@xtl"])),
									  int(float(box["@ybr"])), int(float(box["@xbr"]))])
				frames_boxes[frame].append(box_array)
				frames_labels[frame].append(label)

		return frames_boxes, width, height, frames_labels

		# print(root["track"][0]["box"][0]["@frame"])

def restore_normalization(box, height, width):
	"""
	Scales a normalized box one according to given height and width
	:param box: numpy array of size 4 with [y_min, x_min, y_max, x_max] format
	:param height: height of a picture to which box should be scaled
	:param width: width of a picture to which box should be scaled
	:return: a box scaled according to given height and width
	"""
	norm_box = [int(box[0] * height), int(box[1] * width),
				int(box[2] * height), int(box[3] * width)]

	return np.array(norm_box)

def normalize_box(box, height, width):
	"""
	Normalizes a box according to given height and width
	:param box: numpy array of size 4 with [y_min, x_min, y_max, x_max] format
	:param height: height of a picture from which the box comes
	:param width: width of a picture from which the box comes
	:return: a box normalized according to given height and width
	"""

	norm_box = [box[0] / height, box[1] / width,
				box[2] / height, box[3] / width]

	return np.array(norm_box)

def IoU(box1: np.ndarray, box2: np.ndarray):
	"""
	calculate intersection over union cover percent
	:param box1: box1 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
	:param box2: box2 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
	:return: IoU ratio if intersect, else 0
	"""
	# first unify all boxes to shape (N,4)
	if box1.shape[-1] == 2 or len(box1.shape) == 1:
		box1 = box1.reshape(1, 4) if len(box1.shape) <= 2 else box1.reshape(box1.shape[0], 4)
	if box2.shape[-1] == 2 or len(box2.shape) == 1:
		box2 = box2.reshape(1, 4) if len(box2.shape) <= 2 else box2.reshape(box2.shape[0], 4)
	point_num = max(box1.shape[0], box2.shape[0])
	b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]

	# mask that eliminates non-intersecting matrices
	base_mat = np.ones(shape=(point_num,))
	base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
	base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)

	# I area
	intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1)
	# U area
	union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area
	# IoU
	intersect_ratio = intersect_area / union_area

	return base_mat * intersect_ratio

def calculate_tp_fp_fn(frame_boxes, bench_boxes, bench_w, bench_h, frame_labels = None, bench_labels = None, IoU_threshold = IOU_THRESHOLD):
	detection_TP = 0
	detection_FP = 0
	detection_FN = 0

	recognition_TP = 0
	recognition_FP = 0

	for bench_box in bench_boxes:
		bench_found = False
		for box in frame_boxes:
			if IoU(restore_normalization(box, bench_h, bench_w), bench_box) > IoU_threshold:
				bench_found = True
		if not bench_found:
			detection_FN += 1

	for i, box in enumerate(frame_boxes):
		is_true = False
		for b, bench in enumerate(bench_boxes):
			if IoU(restore_normalization(box, bench_h, bench_w), bench) > IoU_threshold:
				detection_TP += 1
				if bench_labels:
					if frame_labels[i] == bench_labels[b]:
						recognition_TP += 1
					else:
						recognition_FP += 1
				is_true = True
				break
		if not is_true:
			detection_FP += 1
			recognition_FP += 1

	if frame_labels is not None:
		return recognition_TP, recognition_FP
	else:
		return detection_TP, detection_FP, detection_FN

def traverse_group(default_len, frames_group):
	if not frames_group:
		for i in range(default_len):
			yield i, i

		return None, None

	all_count = 0
	for i, frame_gr in enumerate(frames_group):
		group = 1
		if frame_gr:
			group = frame_gr

		for a in range(group):
			yield(i, all_count)
			all_count += 1

def draw_faces(image, face_boxes, face_names):
	if face_boxes:
		for f in range(len(face_boxes)):
			ymin, xmin, ymax, xmax = face_boxes[f]

			if face_names:
				name = face_names[f][0]
			else:
				name = ""

			vis_util.draw_bounding_box_on_image_array(image,
													  ymin,
													  xmin,
													  ymax,
													  xmax,
													  display_str_list=[name],
													  use_normalized_coordinates=is_normalized(
														  face_boxes[0]))

	return image
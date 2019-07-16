# Part of the code used is given under MIT License
# Copyright (c) 2019 https://github.com/yu4u/age-gender-estimation

from .face_expression_kernel import FaceExpressionKernel
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

from ...model.facial_expression.cnn_pytorch import transforms
from ...model.facial_expression.cnn_pytorch.vgg import VGG
from skimage import io
from skimage.transform import resize
import cv2

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class CNNPytorchKernel(FaceExpressionKernel):
	def __init__(self):
		super().__init__()
		self._model_path = parentDir + "/../model/facial_expression/cnn_pytorch/FER2013_VGG19.t7" # path to weight file (e.g. weights.28-3.73.hdf5)
		# Model parameters
		self._depth = 16 # depth of network
		self._width = 8 # width of network
		self._image_size = 48 # size to which every face will be resized
		self._margin = 0.4 # margin around detected face for age-gender estimation
		self._model = None
		self._class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

		cut_size = 44
		self._transform_test = transforms.Compose([
			transforms.TenCrop(cut_size),
			transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
		])


	def load_model(self):
		self._model = VGG('VGG19')
		checkpoint = torch.load(self._model_path)
		self._model.load_state_dict(checkpoint['net'])
		self._model.cuda()
		self._model.eval()

	@staticmethod
	def rgb2gray(rgb):
		return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

	def preprocess(self, faces_img):
		faces = []

		for face in faces_img:
			gray = self.rgb2gray(face)
			gray = resize(gray, (self._image_size, self._image_size), mode='symmetric').astype(np.uint8)

			img = gray[:, :, np.newaxis]

			img = np.concatenate((img, img, img), axis=2)
			img = Image.fromarray(img)
			inputs = self._transform_test(img)

			faces.append(inputs)

		return faces

	def get_expression(self, faces_img):
		predicted_exps = []

		for face in faces_img:
			ncrops, c, h, w = np.shape(face)

			face = face.view(-1, c, h, w)
			face = face.cuda()
			with torch.no_grad():
				face = Variable(face)
			outputs = self._model(face)

			outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

			_, predicted = torch.max(outputs_avg.data, 0)
			pred_exp = str(self._class_names[int(predicted.cpu().numpy())])

			predicted_exps.append(pred_exp)

		return predicted_exps
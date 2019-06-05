import torch, os
from .face_detector_kernel import FaceDetectorKernel
from ...model.dsfd.face_ssd_infer import SSD

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(fileDir)

class DSFDFaceDetector(FaceDetectorKernel):
	def __init__(self):
		super().__init__()
		self._model_path = parentDir + "/../model/dsfd/dsfd.pth"

	def load_model(self):
		self._device = torch.device("cuda")

		self._net = SSD("Inference")
		self._net.load_state_dict(torch.load(self._model_path))
		self._net.to(self._device).eval()

	def inference(self, image):
		target_size = (600, 600)
		scores, boxes = self._net.detect_on_image(image, target_size, self._device, is_pad=False)

		return scores, boxes
#include "pipeline_elements/pipeline_element.h"

class Data {
public:
    Data();
//private:
};

class DataHandlerElem : public PipelineElement {
public:
    DataHandlerElem() {};
//    void run() {};
};

//class Data:
//	def __init__(self, image_data):
//		self._image_data = image_data
//		self._faces = []

//	def add_face(self, face_image, bounding_box):
//		face = Face(face_image, bounding_box)
//		self._faces.append(face)

//	@property
//	def faces(self):
//		return self._faces

//	@property
//	def image_data(self):
//		if type(self._image_data) == np.ndarray:
//			image_data = self._image_data
//			# self._image_data = None
//			return image_data
//		return None

//	def get_JSON(self):
//		data = {}

//		faces = []

//		for face in self._faces:
//			f = dict()

//			f["face"] = face.get_JSON()

//			# f["bounding_box"]
//			faces.append(f)

//		data["faces"] = faces
//		return data

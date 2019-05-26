#include "data_handler.h"
#include <string>

class VideoData : public Data {
public:
    VideoData() {};
};

class VideoHandlerElem : public DataHandlerElem {
public:
    VideoHandlerElem();
    void run();
    void setPathToVideo(std::string pathToVideo);
private:
    std::string m_pathToVideo;
};

//import av, cv2, abc
//from .data_handler import DataHandlerElem, Data

//class Frame(Data):
//	def __init__(self, pts, image_data):
//		super().__init__(image_data)
//		self._pts = pts

//	def get_JSON(self):
//		data = super().get_JSON()
//		data["pts"] = self._pts

//		return data

//class VideoHandlerElem(DataHandlerElem):
//	def run(self, path_to_video):
//		super().run(path_to_video)

//		container = av.open(path_to_video)
//		# Get video stream
//		stream = container.streams.video[0]
//		self.num_of_images = stream.frames
//		# stream.codec_context.skip_frame = 'NONKEY'

//		for frame in container.decode(stream):
//			image = frame.to_rgb().to_ndarray()
			
//			img_height = image.shape[0]
//			img_width = image.shape[1]

//			if img_width > 640:
//				size_ratio = 640/img_width
//				image = cv2.resize(image, dsize=(int(img_width*size_ratio), int(img_height*size_ratio)), interpolation=cv2.INTER_CUBIC)

//			self._current_frame += 1
//			# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
//			if self._current_frame < self._max_frames:
//				yield Frame(frame.pts, image)
//			else:
//				break

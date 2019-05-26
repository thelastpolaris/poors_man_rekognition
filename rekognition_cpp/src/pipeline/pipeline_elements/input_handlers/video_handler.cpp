#include "video_handler.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <array>

#define PAGE_SIZE 1000

VideoHandlerElem::VideoHandlerElem() {

};

void VideoHandlerElem::run() {
        cv::VideoCapture cap(m_pathToVideo); // open the default camera
//        if(!cap.isOpened())  // check if we succeeded
//            return -1;
        std::array<cv::Mat, PAGE_SIZE> frames;

        while(1){
            for (ulong i = 0; i < PAGE_SIZE; i++) {
                std::cout << i << std::endl;
              cv::Mat frame;
              // Capture frame-by-frame
              cap >> frame;

              // If the frame is empty, break immediately
              if (frame.empty())
                break;

              cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
              frames[i] = frame;

//               Display the resulting frame
              imshow( "Frame", frame );

              // Press  ESC on keyboard to exit
              char c=(char) cv::waitKey(25);
              if(c==27)
                break;
            }
        }
}

void VideoHandlerElem::setPathToVideo(std::string pathToVideo) {
    m_pathToVideo = pathToVideo;
}

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

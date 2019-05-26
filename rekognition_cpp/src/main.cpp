#include "pipeline.h"
#include "video_handler.h"
#include <map>

using namespace std;


int main()
{
    VideoHandlerElem* video_handler = new VideoHandlerElem();
    Pipeline* pipeline = new Pipeline({video_handler});

    video_handler->setPathToVideo("/home/polaris/dev/GSoC2019/ccextractor/rekognition/poors_man_rekognition/uploads/seka.mp4");
    pipeline->run();
}


//#include "opencv2/opencv.hpp"
//#include <iostream>
//#include <vector>

//using namespace std;
//using namespace cv;

//int main(){

//  // Create a VideoCapture object and open the input file
//  // If the input is the web camera, pass 0 instead of the video file name
//  VideoCapture cap(0);

//  // Check if camera opened successfully
//  if(!cap.isOpened()){
//    cout << "Error opening video stream or file" << endl;
//    return -1;
//  }

//  std::vector<Mat> mats;

//  while(1){
//    Mat frame;
//    // Capture frame-by-frame
//    cap >> frame;

//    // If the frame is empty, break immediately
//    if (frame.empty())
//      break;

//    // Display the resulting frame
//    imshow( "Frame", frame );
//    mats.push_back(frame);

//    // Press  ESC on keyboard to exit
//    char c=(char)waitKey(25);
//    if(c==27)
//      break;
//  }

//  // When everything done, release the video capture object
//  cap.release();

//  // Closes all the frames
//  destroyAllWindows();

//  return 0;
//}

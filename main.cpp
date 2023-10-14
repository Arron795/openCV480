#include <opencv2/opencv.hpp>
#include "quickopencv.h"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc,char** argv)
{
	//Mat src = imread("C:/Users/sheyi/Pictures/Saved Pictures/oynn10.jpg");
	//Mat src = imread("C:/Users/sheyi/Pictures/¥Á’’/Œ¢–≈Õº∆¨_20210924222118.jpg");
	Mat src = imread("C:/Users/sheyi/Pictures/temp/∫œ’’4.jpg");
	if (src.empty())
	{
		cout << "could not find the image..." << endl; 
		return -1;
	}
	namedWindow("test", WINDOW_NORMAL);
	imshow("test", src);

	QuickDemo qd;
	//qd.colorSpace_Demo(src);
	//qd.pixel_visit_demo(src);
	//qd.operators_demo(src);
	//qd.tracking_bar_demo(src);
	//qd.key_demo(src);
	//qd.channels_demo(src);
	//qd.inrange_demo(src);
	//qd.random_drawing();
	//qd.polyLine_drawing_demo();
	//qd.mouse_drawing_demo(src);
	//qd.resize_demo(src);
	//qd.flip_demo(src);
	//qd.rotate_demo(src);
	//qd.video_demo1(src);
	//qd.video_demo2(src);
	//qd.video_camera(src);
	//qd.histogram_eq_colorful_demo(src);
	//qd.blur_demo(src);
	//qd.gaussian_blur_demo(src);
	//qd.bifilter_demo(src);
	qd.face_detect_demo();
	//qd.face_detection_demo();

	waitKey(0);
	destroyAllWindows();
	return 0;
}
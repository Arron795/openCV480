#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp> // cv::imread
#include <opencv2/objdetect.hpp>  // cv::FaceDetectorYN & cv::FaceRecognizerSF

using namespace cv;

class QuickDemo
{
public:
	//QuickDemo();
	//~QuickDemo();

	void colorSpace_Demo(Mat& image);
	void pixel_visit_demo(Mat& image);
	void operators_demo(Mat& image);
	void tracking_bar_demo(Mat& image);
	void key_demo(Mat& image);
	void channels_demo(Mat& image);
	void inrange_demo(Mat& image);
	void random_drawing();
	void polyLine_drawing_demo();
	void mouse_drawing_demo(Mat& image);
	void resize_demo(Mat& image);
	void flip_demo(Mat& image);
	void rotate_demo(Mat& image);
	void video_demo1(Mat& image);
	void video_demo2(Mat& image);
	int video_key_demo(Mat& image);
	void video_camera(Mat& image);
	void histogram_eq_demo(Mat& image);
	void histogram_eq_colorful_demo(Mat& image);
	void blur_demo(Mat& image);
	void gaussian_blur_demo(Mat& image);
	void bifilter_demo(Mat& image);
	void face_detect_demo();
	void face_detection_demo();


private:

};

//QuickDemo::QuickDemo()
//{
//}
//
//QuickDemo::~QuickDemo()
//{
//}





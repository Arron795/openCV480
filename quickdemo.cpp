#include "quickopencv.h"

void QuickDemo::colorSpace_Demo(Mat& image)
{
	Mat gray, hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("HSV", hsv);
	imshow("�Ҷ�", gray);
	imwrite("D:/repos/openCV/opencv480/img/hsv.png", hsv);
	imwrite("D:/repos/openCV/opencv480/img/gray.png", gray);
}

void QuickDemo::pixel_visit_demo(Mat& image)
{
	int w = image.cols;
	int h = image.rows;
	int dims = image.channels();
	//for (int row = 0; row < h; row++)
	//{
	//	for (int col = 0; col < w; col++)
	//	{
	//		if (dims == 1)	//�Ҷ�ͼ��
	//		{
	//			int pv = image.at<uchar>(row, col);
	//			image.at<uchar>(row, col) = 255 - pv;
	//		}
	//		if (dims == 3)	//��ɫͼ��
	//		{
	//			Vec3b bgr = image.at<Vec3b>(row, col);
	//			image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
	//			image.at<Vec3b>(row, col)[1] = 255 - bgr[1];
	//			image.at<Vec3b>(row, col)[2] = 255 - bgr[2];
	//		}
	//	}
	//}

	for (int row = 0; row < h; row++)
	{
		uchar* current_row = image.ptr<uchar>(row);
		for (int col = 0; col < w; col++)
		{
			if (dims == 1)	//�Ҷ�ͼ��
			{
				int pv = *current_row;
				*current_row++ = 255 - pv;
			}
			if (dims == 3)	//��ɫͼ�� 
			{

				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
			}
		}
	}
	namedWindow("���ض�д��ʾ", WINDOW_FREERATIO);
	imshow("���ض�д��ʾ", image);
}

void QuickDemo::operators_demo(Mat& image)
{
	Mat m = image.zeros(image.size(), image.type());
	Mat dst = image.zeros(image.size(), image.type());
	m = Scalar(50, 50, 50);
	//multiply(image, m, dst);		
	//imshow("�˷�����", dst);

	//�ӷ�����
	//int w = image.cols;
	//int h = image.rows;
	//int dims = image.channels();
	//for (int col = 0; col < w; col++)
	//{
	//	for (int row = 0; row < h; row++)
	//	{
	//		Vec3b p1 = image.at<Vec3b>(row, col);
	//		Vec3b p2 = m.at<Vec3b>(row, col);
	//		dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(p1[0] + p2[0]);	//saturate_cast�����������ж���ֵ�Ƿ����޶���Χ�ڣ��˴�uchar��ָ0~255
	//		dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(p1[1] + p2[1]);
	//		dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(p1[2] + p2[2]);
	//	}
	//}
	add(image, m, dst);
	namedWindow("�ӷ�����", WINDOW_NORMAL);
	imshow("�ӷ�����", dst);
}



static void on_track(int b, void* userdata)
{
	Mat image = *(Mat*)userdata;
	Mat m, dst;
	m = image.zeros(image.size(), image.type());
	dst = image.zeros(image.size(), image.type());
	m = Scalar(b, b, b);
	add(image, m, dst);	//���ȼӷ�
	//subtract(image, m, dst);	//���ȼ���
	imshow("���ȵ���", dst);
}

static void on_lightness(int b, void* userdata)
{
	Mat image = *(Mat*)userdata;
	Mat m, dst;
	m = image.zeros(image.size(), image.type());
	dst = image.zeros(image.size(), image.type());
	addWeighted(image, 1.0, m, 0, b, dst);
	imshow("������Աȶȵ���", dst);
}

static void on_contrast(int b, void* userdata)
{
	Mat image = *(Mat*)userdata;
	Mat m, dst;
	m = image.zeros(image.size(), image.type());
	dst = image.zeros(image.size(), image.type());
	double contrast = b / 100.0;
	addWeighted(image, contrast, m, 0.0, 0, dst);
	imshow("������Աȶȵ���", dst);
}

void QuickDemo::tracking_bar_demo(Mat& image)
{
	namedWindow("������Աȶȵ���", WINDOW_AUTOSIZE);
	int max_value = 100;
	int lightness = 50;
	int contrast_vlaue = 100;
	createTrackbar("Value Bar: ", "������Աȶȵ���", &lightness, max_value, on_lightness, (void*)(&image));
	createTrackbar("Contrast Bar: ", "������Աȶȵ���", &contrast_vlaue, 200, on_contrast, (void*)(&image));
	on_lightness(50, &image);
	on_contrast(100, &image);
}

void QuickDemo::key_demo(Mat& image)
{
	Mat dst, m;
	image.copyTo(dst);
	image.copyTo(m);
	while (true)
	{
		int c = waitKey(100);
		if (c == 27)		//Esc
		{
			break;
		}
		if (c == 49)		//#1
		{
			std::cout << "key 1 pressed." << std::endl;
			cvtColor(image, dst, COLOR_BGR2HSV);
		}
		if (c == 50)		//#2
		{
			std::cout << "key 2 pressed." << std::endl;
			cvtColor(image, dst, COLOR_BGR2GRAY);
		}
		if (c == 51)		//#3
		{
			std::cout << "key 3 pressed." << std::endl;
			m = Scalar(50, 50, 50);
			add(dst, m, dst);
		}
		if (c == 52)		//#4
		{
			std::cout << "key 4 pressed." << std::endl;
			m = Scalar(50, 50, 50);
			subtract(dst, m, dst);
		}
		if (c == 57)		//#9
		{
			std::cout << "key 9 pressed." << std::endl;
			add(image, 0, dst);
		}
		imshow("������Ӧ", dst);
	}
}

void QuickDemo::channels_demo(Mat& image)
{
	std::vector<Mat> mv;
	split(image, mv);		//ͨ������
	imshow("��ɫ", mv[0]);
	imshow("��ɫ", mv[1]);
	imshow("��ɫ", mv[2]);

	Mat dst;
	//mv[0] = 1;
	mv[1] = 0;
	mv[2] = 0;
	merge(mv, dst);			//ͨ���ϲ�
	imshow("���", dst);

	int from_to[] = { 0,2,1,1,2,0 };
	mixChannels(&image, 1, &dst, 1, from_to, 3);
	imshow("ͨ�����", dst);
}

void QuickDemo::inrange_demo(Mat& image)
{
	Mat hsv;
	cvtColor(image, hsv, COLOR_RGB2HSV);
	Mat mask = image.zeros(image.size(), image.type());
	//inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask); //��ɫ
	inRange(hsv, Scalar(100, 43, 46), Scalar(124, 255, 255), mask); //��ɫ


	Mat blueBack = image.zeros(image.size(), image.type());
	blueBack = Scalar(128, 50, 50);
	//bitwise_not(mask, mask);
	imshow("ͼ��ɫ��ת��", mask);
	imshow("��ɫ����", blueBack);
	image.copyTo(blueBack, mask);
	imshow("�滻", blueBack);
}

void QuickDemo::random_drawing()
{
	Mat canvas = Mat::zeros(512, 512, CV_8UC3);
	RNG rng(12345);
	int w = canvas.cols;
	int h = canvas.rows;
	while (true)
	{
		int c;
		c = waitKey(100);
		if (c == 27)
		{
			break;
		}
		int x1 = rng.uniform(0, w);
		int y1 = rng.uniform(0, h);
		int x2 = rng.uniform(0, w);
		int y2 = rng.uniform(0, h);
		Point pt1(x1, y1);
		Point pt2(x2, y2);
		int b = rng.uniform(0, 255);
		int g = rng.uniform(0, 255);
		int r = rng.uniform(0, 255);
		canvas = Scalar(0, 0, 0);
		line(canvas, pt1, pt2, Scalar(b, g, r), 1, LINE_AA, 0);
		imshow("�漴����", canvas);
	}
}

void QuickDemo::polyLine_drawing_demo()
{
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
	Point p1(100, 100);
	Point p2(25, 22);
	Point p3(135, 12);
	Point p4(22, 231);
	std::vector<Point> pts;
	pts.push_back(p1);
	pts.push_back(p2);
	pts.push_back(p3);
	pts.push_back(p4);

	//fillPoly(canvas, pts, Scalar(100, 0, 100), 8, 0);
	//polylines(canvas, pts, true, Scalar(255, 0, 0), 1, 32, 0);
	std::vector<std::vector<Point>> contours;
	contours.push_back(pts);

	drawContours(canvas, contours, -1, Scalar(255, 0, 0), 2);
	imshow("���", canvas);

}

Point sp(-1, -1);
Point ep(-1, -1);
Mat temp;
static void on_draw(int event, int x, int y, int flags, void* userdata)
{
	Mat image = *((Mat*)userdata);
	if (event == EVENT_LBUTTONDOWN)
	{
		sp.x = x;
		sp.y = y;
		std::cout << "���������£�����Ϊ " << sp << std::endl;
	}
	else if (event == EVENT_LBUTTONUP)
	{
		ep.x = x;
		ep.y = y;
		std::cout << "������̧������Ϊ " << ep << std::endl;
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;
		if (dx > 0 && dy > 0)
		{
			Rect box(sp.x, sp.y, dx, dy);
			imshow("ROI����", image(box));
			Point center(sp.x + dx / 2, sp.y + dy / 2);
			//circle(image, center,dx/2,Scalar(255,0,0),2,8,0 );//Բ��
			Size axes(dx / 2, dy / 2);	//����Ͷ���
			ellipse(image, center, axes, 0.0, 0.0, 360.0, Scalar(255, 0, 0), 2, 8, 0);
			imshow("������", image);
			sp.x = -1;
			sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		if (sp.x > 0 && sp.y > 0)
		{
			ep.x = x;
			ep.y = y;
			std::cout << "����ƶ�����ʱ����Ϊ " << ep << std::endl;
			int dx = ep.x - sp.x;
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0)
			{
				temp.copyTo(image);
				Rect box(sp.x, sp.y, ep.x, ep.y);
				Point center(sp.x + dx / 2, sp.y + dy / 2);
				//circle(image, center,dx/2,Scalar(255,0,0),2,8,0 );//Բ��
				Size axes(dx / 2, dy / 2);	//����Ͷ���
				ellipse(image, center, axes, 0.0, 0.0, 360.0, Scalar(255, 0, 0), 2, 8, 0);	//��Բ
				imshow("������", image);
			}
		}
	}
}

void QuickDemo::mouse_drawing_demo(Mat& image)
{
	namedWindow("������", WINDOW_NORMAL);
	imshow("������", image);
	setMouseCallback("������", on_draw, (void*)(&image));
	temp = image.clone();
}

void QuickDemo::resize_demo(Mat& image)
{
	Mat zoomin, zoomout;
	int w = image.rows;
	int h = image.cols;
	resize(image, zoomin, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);
	resize(zoomin, zoomout, Size(w * 1.5, h * 1.5), 0, 0, INTER_LINEAR);
	imshow("zoomin", zoomin);
	imshow("zoomout", zoomout);
}

void QuickDemo::flip_demo(Mat& image)
{
	Mat dst;

	flip(image, dst, 0);
	imshow("��x�ᷭת", dst);
	flip(image, dst, 1);
	imshow("��y�ᷭת", dst);
	flip(image, dst, -1);
	imshow("��x���y�ᷭת", dst);
}

void QuickDemo::rotate_demo(Mat& image)
{
	Mat dst, M;
	int w = image.cols;
	int h = image.rows;
	Point2f center(w / 2, h / 2);
	M = getRotationMatrix2D(center, 45, 1.0);
	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));
	int nw = cos * w + sin * h;
	int nh = cos * h + sin * w;
	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	warpAffine(image, dst, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(255, 255, 255));
	namedWindow("��תչʾ", WINDOW_NORMAL);
	imshow("��תչʾ", dst);
}

void QuickDemo::video_demo1(Mat& image)
{
	//VideoCapture vc(0);
	VideoCapture vc("D:/Videos/è������CD9-��СѼ.avi");
	Mat frame;
	Mat dst, m;
	while (true)
	{
		vc.read(frame);
		frame.copyTo(dst);
		frame.copyTo(m);
		//flip(frame,frame,1);
		int res = video_key_demo(frame);
		if (frame.empty())
		{
			break;
		}
		else
		{
			if (res == 1)
			{
				cvtColor(frame, dst, COLOR_BGR2HSV);
			}
			if (res == 2)
			{
				cvtColor(frame, dst, COLOR_BGR2GRAY);
			}
			if (res == 3)
			{
				cvtColor(frame, dst, 0);
			}
			if (res == 4)
			{
				m = Scalar(50, 50, 50);
				addWeighted(frame, 1.0, m, 1, 0.0, dst);
			}

		}
		imshow("����", dst);
		imshow("��Ƶ", frame);
		//TODO: do something...
		int c = waitKey(10);
		if (c == 27)
		{
			break;
		}
	}
	vc.release();
}

void QuickDemo::video_demo2(Mat& image)
{
	VideoCapture vc;
	vc.open("D:/Videos/MyVideo/sky.mp4");
	int frame_width = vc.get(CAP_PROP_FRAME_WIDTH);
	int frame_heght = vc.get(CAP_PROP_FRAME_HEIGHT);
	int count = vc.get(CAP_PROP_FRAME_COUNT);		//֡��			֡��/֡�� = ����ʱ��
	double fps = vc.get(CAP_PROP_FPS);			//֡�� ��֡ÿ�룩

	std::cout << "frame_width: " << frame_width << std::endl;
	std::cout << "frame_heght: " << frame_heght << std::endl;
	std::cout << "count(number of frame): " << count << std::endl;
	std::cout << "fps: " << fps << std::endl;

	VideoWriter	writer("D:/Videos/MyVideo/sky_test.mp4", CAP_PROP_FOURCC, fps, Size(frame_width, frame_heght));
	Mat frame;
	while (true)
	{
		vc.read(frame);
		waitKey(10);
		if (frame.empty())
		{
			break;
		}
		imshow("videoDemo2", frame);
		writer.write(frame);
	}
	vc.release();
	writer.release();
}

int QuickDemo::video_key_demo(Mat& image)
{

	int c = waitKey(10);
	if (c == 49)
	{
		std::cout << "key #1 pressed.." << std::endl;
		return 1;
	}
	if (c == 50)
	{
		std::cout << "key #2 pressed.." << std::endl;
		return 2;
	}
	if (c == 51)
	{
		std::cout << "key #3 pressed.." << std::endl;
		return 3;
	}
	if (c == 52)
	{
		std::cout << "key #4 pressed.." << std::endl;
		return 4;
	}
	if (c == 27)
	{
		std::cout << "key #ESC pressed.." << std::endl;
		return 0;
	}
}

void QuickDemo::video_camera(Mat& image)
{
	VideoCapture vc;
	vc.open("D:/Videos/MyVideo/sky.mp4");
	Mat frame;
	int frame_height = vc.get(CAP_PROP_FRAME_HEIGHT);
	int frame_width = vc.get(CAP_PROP_FRAME_HEIGHT);
	double fps = vc.get(CAP_PROP_FPS);
	int count = vc.get(CAP_PROP_FRAME_COUNT);
	int format = vc.get(CAP_PROP_FORMAT);
	int fourcc = vc.get(CAP_PROP_FOURCC);

	std::cout << "frame_width: " << frame_width << std::endl;
	std::cout << "frame_heght: " << frame_height << std::endl;
	std::cout << "count(number of frame): " << count << std::endl;
	std::cout << "fps: " << fps << std::endl;
	std::cout << "format: " << format << std::endl;
	std::cout << "fourcc: " << fourcc << std::endl;
	while (true)
	{
		vc.read(frame);
		int c = waitKey(10);
		if (c == 27 || frame.empty())
		{
			break;
		}
		imshow("Camera", frame);
	}
	vc.release();

}


void QuickDemo::histogram_eq_demo(Mat& image)
{
	Mat gray, dst;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("���⻯ǰ", gray);
	equalizeHist(gray, dst);		//ֱ��ͼ���⻯ ֻ֧�ֵ�ͨ��
	imshow("���⻯��", dst);
}

void QuickDemo::histogram_eq_colorful_demo(Mat& image)
{
	Mat gray, dst;
	std::vector<Mat> mv;
	split(image, mv);

	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("���⻯ǰ", gray);

	for (int i = 0; i < 3; i++)
	{
		equalizeHist(mv[i], mv[i]);		//��ÿ��ͨ������ֱ��ͼ���⻯		//��ͨ��������ֻ�л�ɫ
	}

	merge(mv, dst);				//merge(1.Mat��vector������2.Ŀ��ͼ��)

	imshow("���⻯��", dst);
}

void QuickDemo::blur_demo(Mat& image)
{
	Mat dst;
	blur(image, dst, Size(5, 5), Point(-1, -1));
	imshow("ͼ��ģ��", dst);
}

void QuickDemo::gaussian_blur_demo(Mat& image)
{
	Mat dst;
	GaussianBlur(image, dst, Size(0, 0), 15);
	imshow("��˹ģ��", dst);
}

void QuickDemo::bifilter_demo(Mat& image)
{
	//VideoCapture vc;
	//vc.open(0);
	//Mat frame,dst;
	//while (true)
	//{
	//	int c = waitKey(10);
	//	vc.read(frame);  
	//	bilateralFilter(frame, dst, 0, 100, 10);
	//	if (frame.empty() || c == 27 )
	//	{
	//		break;
	//	}
	//	imshow("˫��ģ��ǰ", frame);
	//	imshow("˫��ģ����",dst);
	//}
	//vc.release();
	Mat dst;
	bilateralFilter(image, dst, 0, 100, 10);
	imshow("˫��ģ��", dst);
}

//���ڲ�����...
void QuickDemo::face_detect_demo()
{
	String modlePath = "D:/openCV/opencv/sources/samples/dnn/face_detector/yolov2-tiny.weights/";
	String configPath = "D:/openCV/opencv/sources/samples/dnn/face_detector/yolov2-tiny.cfg";
	Mat image = imread("C:/Users/sheyi/Pictures/temp/����4.jpg");
	Ptr<FaceDetectorYN> faceDetector = FaceDetectorYN::create(modlePath, configPath, image.size());
	Mat faces;
	faceDetector->detect(image, faces);
}

void QuickDemo::face_detection_demo()
{
	String root_dir = "D:/openCV/opencv/sources/samples/dnn/face_detector/";
	dnn::Net net = dnn::readNetFromTensorflow(
		root_dir + "opencv_face_detector_uint8.pb",
		root_dir + "opencv_face_detector.pbtxt");
	VideoCapture capture("C:/Users/sheyi/Pictures/temp/����4.jpg");
	Mat frame;
	namedWindow("�������", WINDOW_NORMAL);
	while (true)
	{
		capture.read(frame);
		if (frame.empty())
		{
			break;
		}
		Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);		//��ȡģ��
		//TODO
		net.setInput(blob);			//׼������		NCHW  ����ͼ ����ͨ�� ���
		Mat probs = net.forward();	//�������
		Mat detectionMat(probs.size[2],probs.size[3],CV_32F, probs.ptr<float>());
		//�������
		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > 0.5)
			{
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
				Rect box(x1, y1, x2 - x1, y2 - y1);
				rectangle(frame, box, Scalar(0, 255, 0), 2, 8, 0);
			}
		}
		imshow("�������", frame);
		int c = waitKey(10);
		if (c == 27)
		{
			break;
		}
	}
	capture.release();
}



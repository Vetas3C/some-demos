#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <wiringPi.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <wiringSerial.h>
#include <iostream>
#include <iomanip> 
#include <sstream>  


using namespace cv;
using namespace std;

void serial_send_data(int serial, string red_x, string red_y, string green_x, string green_y);
string toFourDigitString(int number);

// 红色 HSV 阈值范围
const Scalar LOWER_RED1 = Scalar(0, 120, 70);
const Scalar UPPER_RED1 = Scalar(10, 255, 255);
const Scalar LOWER_RED2 = Scalar(160, 120, 70);
const Scalar UPPER_RED2 = Scalar(180, 255, 255);
// HSV范围为绿色 =====
const Scalar lower_green(35, 50, 50); 
const Scalar upper_green(85, 255, 255); 

Point2f reddot_center;
Point2f greendot_center;

int fd ;

int main() {
	if ((fd = serialOpen ("/dev/ttyAS5", 115200)) < 0)
	{
		fprintf (stderr, "Unable to open serial device: %s\n", strerror (errno)) ;
		return 1 ;
	}
	
	if (wiringPiSetup () == -1)
	{
		fprintf (stdout, "Unable to start wiringPi: %s\n", strerror (errno)) ;
		return 1 ;
	}
	
	
    // 打开摄像头
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "无法打开摄像头！" << endl;
        return -1;
    }

    // 创建窗口
    //namedWindow("Red Dot Detection", WINDOW_AUTOSIZE);

    Mat frame, hsv, redmask1, redmask2, redmask, greenmask;
    vector<vector<Point>> red_contours,green_contours;

    while (true) 
	{
        // 读取帧
        cap >> frame;
        if (frame.empty()) break;

        // 转换为HSV颜色空间
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // 创建红色掩膜
        inRange(hsv, LOWER_RED1, UPPER_RED1, redmask1);
        inRange(hsv, LOWER_RED2, UPPER_RED2, redmask2);
        bitwise_or(redmask1, redmask2, redmask);
		// 创建绿色掩膜
		inRange(hsv, lower_green, upper_green, greenmask);

        // 形态学操作（去除噪声）
        Mat redkernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(redmask, redmask, MORPH_OPEN, redkernel);
		
		Mat greenkernel = getStructuringElement(MORPH_ELLIPSE, Size(7,7));
        morphologyEx(greenmask, greenmask, MORPH_OPEN, greenkernel);

        

        // 查找轮廓
        findContours(redmask, red_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		findContours(greenmask, green_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // 绘制结果
        if (!red_contours.empty() && !green_contours.empty()) 
		{
            // 找到最大轮廓
            auto RedmaxContour = *max_element(red_contours.begin(), red_contours.end(),
                [](const vector<Point>& a, 
				const vector<Point>& b) {return contourArea(a) < contourArea(b);});
				
			auto GreenmaxContour = *max_element(green_contours.begin(), green_contours.end(),
                [](const vector<Point>& c, 
				const vector<Point>& d) {return contourArea(c) < contourArea(d);});

            // 计算找到的最大轮廓的最小外接圆
            Point2f red_center;
            float red_radius;
			Point2f green_center;
            float green_radius;
            minEnclosingCircle(RedmaxContour, red_center, red_radius);
            minEnclosingCircle(GreenmaxContour, green_center, green_radius);
			
			

            // 绘制结果
            if (red_radius > 5) // 过滤小噪点
			{ 
                //circle(frame, red_center, red_radius, Scalar(0, 255, 0), 2);
                //circle(frame, red_center, 2, Scalar(0, 255, 0), -1);
                //putText(frame, 
                //        format("Position: (%.0f, %.0f)", red_center.x, red_center.y),
                //        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7,
                //        Scalar(0, 255, 0), 2);
				reddot_center.x = red_center.x;
				reddot_center.y = red_center.y;
				cout << "redpoint: x = " << reddot_center.x << ", y = " << reddot_center.y <<endl;
            }
			
            if (green_radius > 5) // 过滤小噪点
			{ 
                //circle(frame, green_center, green_radius, Scalar(0, 255, 0), 2);
                //circle(frame, green_center, 2, Scalar(0, 255, 0), -1);
                //putText(frame, 
                //        format("Position: (%.0f, %.0f)", green_center.x, green_center.y),
                //        Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7,
                //        Scalar(0, 255, 0), 2);
				greendot_center.x = green_center.x;
				greendot_center.y = green_center.y;
				cout << "greenpoint: x = " << greendot_center.x << ", y = " << greendot_center.y <<endl;
            }
			
			string redx = toFourDigitString(reddot_center.x);
			string redy = toFourDigitString(reddot_center.y);
			string greenx = toFourDigitString(greendot_center.x);
			string greeny = toFourDigitString(greendot_center.y);
			
			serial_send_data(fd, redx, redy, greenx, greeny);
        }

        // 显示结果
        //imshow("Red And Green Dot Detection", frame);

        // 退出条件
        if (waitKey(200) == 27) break; // ESC键退出
    }

    // 释放资源
    cap.release();
    destroyAllWindows();
	close(fd);
    return 0;
}

void serial_send_data(int serial, string red_x, string red_y, string green_x, string green_y)
{
	//char *sendBuffer = (char *)malloc(64);
	//memset(sendBuffer,'\0',sizeof(sendBuffer));
	//sprintf(sendBuffer,"%s %s %s %s!",red_x.c_str(),red_y.c_str(),green_x.c_str(),green_y.c_str());
	string send_str = red_x + " " + red_y + " " + green_x + " " + green_y + "!";
	serialPuts(serial,send_str.c_str());
}

string toFourDigitString(int number) {
    std::ostringstream oss;
    oss << std::setw(4) << std::setfill('0') << number;
    return oss.str();
}

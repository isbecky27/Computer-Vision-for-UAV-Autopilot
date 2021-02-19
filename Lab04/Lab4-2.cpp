/*  LAB4-2  */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <windows.h>
using namespace cv;
using namespace std;

int main()
{
    VideoCapture cap;
    cap.open(1);

    Mat img = imread("warp.jpg");
    Mat frame;

    while (cap.isOpened()) {
        
        cap >> frame;

        int frame_height = frame.rows;
        int frame_width = frame.cols;

        vector<Point2f> corners(4);
        corners[0] = Point2f(0, 0);
        corners[1] = Point2f(0, frame_height - 1);
        corners[2] = Point2f(frame_width - 1, frame_height - 1);
        corners[3] = Point2f(frame_width - 1, 0);
        vector<Point2f> corners_trans(4);
        corners_trans[0] = Point2f(194, 114);
        corners_trans[1] = Point2f(178, 274);
        corners_trans[2] = Point2f(456, 275); 
        corners_trans[3] = Point2f(463, 50); 
        vector<Point> point;
        point.push_back(Point(194, 114));  
        point.push_back(Point(178, 274));
        point.push_back(Point(456, 275));
        point.push_back(Point(463, 50));

        Mat transform = getPerspectiveTransform(corners, corners_trans);
        //cout << transform << endl;
        
        Mat dst;
        warpPerspective(frame, dst, transform, img.size(), INTER_LINEAR);

        fillConvexPoly(img, point, Scalar(0,0,0));

        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img.at<Vec3b>(i, j)[0] += dst.at<Vec3b>(i, j)[0];
                img.at<Vec3b>(i, j)[1] += dst.at<Vec3b>(i, j)[1];
                img.at<Vec3b>(i, j)[2] += dst.at<Vec3b>(i, j)[2];
            }
        }

        imshow("result", img);
        waitKey(30);

    }
    
    return 0;
}
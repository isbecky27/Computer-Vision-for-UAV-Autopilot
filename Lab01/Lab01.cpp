#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;


int main() {
	Mat img1 = imread("kobe.jpg");
	Mat img2 = imread("IU.png");

	/** 1-1 **/
	// cvtColor(img, gray, COLOR_RGB2GRAY); 

	Mat gray(img1.rows, img1.cols, CV_8U, Scalar(200));

	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			int rgb = 0;
			for(int k=0;k<3;k++){
				rgb += img1.at<Vec3b>(i, j)[k];
			}
			gray.at<uchar>(i, j) = rgb / 3;
		}
	}

	imwrite("kobe_gray.jpg", gray);
	//imshow("gray", gray);
	//waitKey(0);
	
	/** 1-2 **/ 
	//resize(img2, zoom_in, Size(img2.cols * 3, img2.rows *3), 0, 0, INTER_NEAREST);
	float zoom = 3;
	Mat  zoom_in(img2.rows*zoom, img2.cols*zoom, CV_8UC3, Scalar(200));

	for (int i = 0; i < zoom_in.rows; i++) {
		for (int j = 0; j < zoom_in.cols; j++) {
			
			zoom_in.at<Vec3b>(i, j)[0] = img2.at<Vec3b>(i/zoom, j/zoom)[0];
			zoom_in.at<Vec3b>(i, j)[1] = img2.at<Vec3b>(i / zoom, j / zoom)[1];
			zoom_in.at<Vec3b>(i, j)[2] = img2.at<Vec3b>(i / zoom, j / zoom)[2];

		}
	}

	imwrite("IU_3.png", zoom_in);
	//imshow("zoom_in", zoom_in);
	//waitKey(0);

	/** 1-3 **/ 
	//resize(img2, zoom_out, Size(img2.cols * 0.7, img2.rows *0.7), 0, 0, INTER_LINEAR);
	
	zoom = 0.7;
	Mat  zoom_out(img2.rows * zoom, img2.cols * zoom, CV_8UC3, Scalar(200));

	for (int i = 0; i < zoom_out.rows; i++) {
		for(int j = 0; j < zoom_out.cols; j++){

			float x = i *(1/zoom);
			float y = j*(1/zoom);

			int ii = x / 1;
			int jj = y / 1;
			float u = x - ii;
			float v = y - jj;

			if (x == ii && y == jj) {
				zoom_out.at<Vec3b>(i, j)[0] = img2.at<Vec3b>(ii, jj)[0];
				zoom_out.at<Vec3b>(i, j)[1] = img2.at<Vec3b>(ii, jj)[1];
				zoom_out.at<Vec3b>(i, j)[2] = img2.at<Vec3b>(ii, jj)[2];

				continue;
			}
			
			for (int k = 0; k < 3; k++) {
				float ans ;
				ans =	(1 - u) * (1 - v) * img2.at<Vec3b>(ii, jj)[k] + (1 - u) * v * img2.at<Vec3b>(ii, jj + 1)[k] + u * (1 - v) * img2.at<Vec3b>(ii + 1, jj)[k] + u * v * img2.at<Vec3b>(ii + 1, jj + 1)[k];			
				if ((ans - int(ans )) >= 0.5) {
					zoom_out.at<Vec3b>(i, j)[k] = int(ans) + 1;
				}
				else {
					zoom_out.at<Vec3b>(i, j)[k] = int(ans);
				}
				
			}
			
		}
	}

	imwrite("IU_0.7.png", zoom_out);
	imshow("zoom_out", zoom_out);
	waitKey(0);
	
	return 0;
}


#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;


int main() {
	Mat img = imread("mj.tif",0);
	Mat img2 = imread("HyunBin.jpg", 0);

	 /** 2-1 **/

	float arr[256][2];
	int total = img.cols * img.rows;
	Mat gray(img.rows, img.cols, CV_8U, Scalar(200));

	for (int i = 0; i < 256; i++) {
		arr[i][0] = 0;
	}
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			arr[img.at<uchar>(i, j)] [0]+= 1;
		}
	}

	for (int i = 0; i < 256; i++) {
		if (i == 0) {
			arr[i][1] = arr[i][0] / float(total);
			continue;
		}
		arr[i][1] = arr[i][0] /float( total)+ arr[i-1][1];
		//cout << i << " " << arr[i][1] << endl;
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			gray.at<uchar>(i, j) = int(arr[img.at<uchar>(i, j)][1]*256);
		}
	}

	imwrite("mj_histogram.tif", gray);
	imshow("histogram", gray);
	waitKey(0);


	/** 2-2 **/

	float arr2[256][2];
	int total2 = img2.cols * img2.rows;
	Mat gray2(img2.rows, img2.cols, CV_8U, Scalar(200));
	Mat x(img2.rows, img2.cols, CV_8U, Scalar(200));
	Mat y(img2.rows, img2.cols, CV_8U, Scalar(200));
	Mat output(img2.rows, img2.cols, CV_8U, Scalar(200));


	for (int i = 0; i < 256; i++) {
		arr2[i][0] = 0;
	}

	for (int i = 0; i < img2.rows; i++) {
		for (int j = 0; j < img2.cols; j++) {
			arr2[img2.at<uchar>(i, j)][0] += 1;
		}
	}

	for (int i = 0; i < 256; i++) {
		if (i == 0) {
			arr2[i][1] = arr2[i][0] / float(total2);
			continue;
		}
		arr2[i][1] = arr2[i][0] / float(total2) + arr2[i - 1][1];
	}

	for (int i = 0; i < img2.rows; i++) {
		for (int j = 0; j < img2.cols; j++) {
			gray2.at<uchar>(i, j) = int(arr2[img2.at<uchar>(i, j)][1] * 256);
		}
	}


	for (int i = 1; i < img2.rows-1; i++) {
		for (int j = 1; j < img2.cols - 1; j++) {
			int temp = gray2.at<uchar>(i - 1, j - 1) * (-1) + gray2.at<uchar>(i, j - 1) * 0 + gray2.at<uchar>(i + 1, j - 1) * 1 +
				gray2.at<uchar>(i - 1, j) * (-2) + gray2.at<uchar>(i, j) * 0 + gray2.at<uchar>(i + 1, j) * 2 +
				gray2.at<uchar>(i - 1, j + 1) * (-1) + gray2.at<uchar>(i, j + 1) * 0 + gray2.at<uchar>(i + 1, j + 1) * 1;

			if (temp <= 255 && temp >= 120) {
				x.at<uchar>(i, j) = 255;
			}
			else {
				x.at<uchar>(i, j) = 0;
			}
		}
	}

	imwrite("HyunBin_x.jpg", x);
	imshow("maskx", x);
	waitKey(0);

	
	for (int i = 1; i < img2.rows - 1; i++) {
		for (int j = 1; j < img2.cols - 1; j++) {
			int temp = gray2.at<uchar>(i - 1, j - 1) * (-1) + gray2.at<uchar>(i, j - 1) * (-2) + gray2.at<uchar>(i + 1, j - 1) * (-1) +
				gray2.at<uchar>(i - 1, j) * 0 + gray2.at<uchar>(i, j) * 0 + gray2.at<uchar>(i + 1, j) * 0 +
				gray2.at<uchar>(i - 1, j + 1) * 1 + gray2.at<uchar>(i, j + 1) * 2 + gray2.at<uchar>(i + 1, j + 1) * 1;

			if (temp <= 255 && temp >= 120) {
				y.at<uchar>(i, j) = 255;
			}
			else {
				y.at<uchar>(i, j) = 0;
			}
		}
	}

	imwrite("HyunBin_y.jpg", y);
	imshow("masky", y);
	waitKey(0);

	
	for (int i = 1; i < img2.rows - 1; i++) {
		for (int j = 1; j < img2.cols - 1; j++) {
				int temp=x.at<uchar>(i,j)+y.at<uchar>(i, j) ;
				if (temp >= 255) {
					output.at<uchar>(i, j) = 255;
				}
				else output.at<uchar>(i, j) = 0;

		}
	}

	imwrite("HyunBin_output.jpg", output);
	imshow("output",output);
	waitKey(0);

	return 0;
}


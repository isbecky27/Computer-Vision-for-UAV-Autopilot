/*  LAB 3-2  */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

int main() {

	RNG rng;
	Mat img = imread("output.jpg", 0);
	Mat gray(img.rows, img.cols, CV_16U, Scalar(0));

	int flag = 1;

	vector <int > vec;
	vec.push_back(0);

	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			
			if (img.at<uchar>(j, i) <= 120) {
				//gray.at<short>(j, i) = 0;
				continue;
			}

			if (i == 0 && j == 0) {
				gray.at<short>(j, i) = flag;
				vec.push_back(flag);
				flag++;
			}
			else if (j == 0) {
				if (gray.at<short>(j,i-1)!=0) {
					gray.at<short>(j, i) = gray.at<short>(j, i-1);
				}
				else {
					gray.at<short>(j, i) = flag;
					vec.push_back(flag);
					flag++;
				}
			}
			else if ( i == 0) {
				if (gray.at<short>(j-1, i) != 0) {
					gray.at<short>(j, i) = gray.at<short>(j-1, i);
				}
				else {
					gray.at<short>(j, i) = flag;
					vec.push_back(flag);
					flag++;
				}
			}
			else {
				if (gray.at<short>(j, i-1) == 0 && gray.at<short>(j-1, i) == 0) {
					gray.at<short>(j, i) = flag;
					vec.push_back(flag);
					flag++;
				}
				else if(gray.at<short>(j, i-1) == 0 && gray.at<short>(j-1, i) != 0){
					gray.at<short>(j, i) = gray.at<short>(j-1, i);
				}
				else if (gray.at<short>(j, i-1) != 0 && gray.at<short>(j-1, i) == 0) {
					gray.at<short>(j, i) = gray.at<short>(j, i-1);
				}
				else {
					if (gray.at<short>(j, i-1) == gray.at<short>(j-1, i)) {
						gray.at<short>(j, i) = gray.at<short>(j - 1, i);
					}
					else if (gray.at<short>(j, i-1) > gray.at<short>(j-1, i)) {
						gray.at<short>(j, i) = gray.at<short>(j - 1, i);
						vec[gray.at<short>(j, i - 1)] = gray.at<short>(j - 1, i);
					}
					else {
						 gray.at<short>(j, i) = gray.at<short>(j, i-1);
						 vec[gray.at<short>(j-1, i)] = gray.at<short>(j, i-1);
					}
				}
			
			}
		}
	}

	//imshow("Gray", gray);
	//waitKey(0);

	int arr[3][300];

	for (int i = 0; i <3; i++) {
		for (int j = 0; j < 300; j++) {
			arr[i][j]= rng.uniform(90, 255);
		}
	}
	
	Mat color(img.rows, img.cols, CV_8UC3);
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			if (gray.at<short>(i, j)== 0) {
				color.at<Vec3b>(i, j)[0] = 0;
				color.at<Vec3b>(i, j)[1] = 0;
				color.at<Vec3b>(i, j)[2] = 0;
			}
			else {
				int temp = gray.at<short>(i, j);
				
				while (vec[temp] != temp) {
					temp=vec[temp];
				}
			
				color.at<Vec3b>(i, j)[0] = arr[0][temp%300];
				color.at<Vec3b>(i, j)[1] = arr[1][temp%300]; 
				color.at<Vec3b>(i, j)[2] = arr[2][temp%300];
				//color.at<Vec3b>(i, j)[2] = 100;
				//color.at<Vec3b>(i, j)[1] = 255;
				//color.at<Vec3b>(i, j)[0] = 100;
			}
		}
	}

	imwrite("output2.jpg",color);
	imshow("output", color);
	waitKey(0);

	return 0;
}


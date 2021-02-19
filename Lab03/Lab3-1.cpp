/*  LAB 3-1  */
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {

	Mat img = imread("input.jpg",0);
	Size s = img.size();
	Mat gray = img.clone();
	long long all[256] = {0};
	long long b = 0, o = 0, no=0, nb=0, total=s.width*s.height, result, ans[2];
	double eb, eo,neb,neo;
	for (int j = 0; j < s.width; j++) {
		for (int i = 0; i < s.height; i++) {
			all[img.at<uchar>(i, j)] += 1;
		}
	}
	b = all[0];
	o = total - b;
	eb = 0;
	long long tmp=0;
	for (int i = 1; i < 256; i++) {
		tmp += i * all[i];
	}
	eo = tmp / o;
	result = b * o * pow((eb-eo), 2);
	//cout << result <<" "<<b<<" "<<o<<" "<<eo<<" "<< endl << endl;
	ans[0] = 0; ans[1] = result;
	for (int i = 1; i < 255; i++) {
		nb =b+ all[i];
		no = o-all[i];
		neb = (b * eb + i*all[i]) / nb;
		neo = (o * eo - i*all[i]) / no;
		b = nb; o = no; eb = neb; eo = neo;
		result = b * o * pow((eb - eo), 2);
		//cout << result << endl;
		if (result > ans[1]){
			ans[0] = i; ans[1] = result;
		}
	}
	for (int j = 0; j < s.width; j++) {
		for (int i = 0; i < s.height; i++) {
			if (gray.at<uchar>(i, j) > ans[0]) {
				gray.at<uchar>(i, j) = 255;
			}
			else {
				gray.at<uchar>(i, j) = 0;
			}
		}
	}
	cout << ans[0] << endl;

	imwrite("output.jpg", gray);
	namedWindow("demo");
	imshow("demo", gray);
	waitKey(0);

	return 0;
}
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

void conv2D(Mat, Mat, Mat&, int borderType = BORDER_DEFAULT);
Mat gaussBlur(Mat, Size, float, int borderType = BORDER_DEFAULT);

// edge detection
void prewitt(const Mat&, Mat&, int, int, int borderType = BORDER_DEFAULT);
void scharr(const Mat&, Mat&, int, int, int borderType = BORDER_DEFAULT);
void LoG(Mat, Size, float, int, Mat&);
int factorial(int);
Mat getPascalSmooth(int);
Mat getPascalDiff(int);
Mat sobel(Mat, int, int, int, int borderType = BORDER_DEFAULT);
enum class norm_type { L1, L2 };
Mat calcMagnitude(Mat, Mat, norm_type);

// Canny detector
Mat nonMaxSuppression(Mat, Mat, norm_type);
bool isInRange(int, int, int, int);
void track(Mat, Mat&, float, int, int, int, int);
void CannyDetect(Mat input, Mat& output, float lowerThresh, float upperThresh, norm_type type);


int main(int argc, char** argv) {
	// Read preprocessed images
	Mat images[5];
	for (int i = 1; i <= 5; i++) {
		string fileName("images/img");
		//string fileName("images/p1im");
		fileName = fileName + to_string(i) + ".png";
		images[i-1] = imread(fileName);
		if (!images[i-1].data) {
			cout << "Load image " << i << " failed! Please check and try again." << endl;
			return EXIT_FAILURE;
		}
	}

	Mat results[4][2], img_show[2], show;

	// Change the index to decide the input image
	int index = 0;
	// prewitt
	prewitt(images[index], results[0][0], 0, 1, BORDER_REPLICATE);
	prewitt(images[index], results[0][1], 1, 0, BORDER_REPLICATE);
	
	// scharr
	scharr(images[index], results[1][0], 0, 1, BORDER_REPLICATE);
	scharr(images[index], results[1][1], 1, 0, BORDER_REPLICATE);
	
	// Sobel	
	results[2][0] = sobel(images[index], 0, 1, 3, BORDER_REPLICATE);
	results[2][1] = sobel(images[index], 1, 0, 3, BORDER_REPLICATE);
	
	// LoG
	LoG(images[index], Size(3,3), 2.0, BORDER_REPLICATE, results[3][0]);

	// Canny detection
	//GaussianBlur(images[0], results[0][0], Size(3, 3), 1.0);
	//CannyDetect(images[index], results[0], 40, 200, norm_type::L1);


	Mat r[2];
	/*
	for (int i = 0; i < 3; i++) {
		r[i] = calcMagnitude(results[i][0], results[i][1], norm_type::L1);
	}
	//r[3] = cv::abs(results[3][0]);
	*/

	//r[0] = calcMagnitude(results[0][0], results[0][1], norm_type::L1);
	//r[1] = calcMagnitude(results[1][0], results[1][1], norm_type::L1);
	
	//hconcat(images[index], results[index], img_show);
	//hconcat(results[1][1], results[2][1], show);
	
	for (int i = 0; i < 4; i++) {
		resize(results[i], results[i], Size(360, 220));
		//cvtColor(r[i], r[i], COLOR_BGR2GRAY);
		//threshold(r[i], r[i], 100 , 255, THRESH_BINARY);
	}
	
	//hconcat(r[0], r[1], img_show[0]);
	//hconcat(r[2], r[3], img_show[1]);
	hconcat(results[0], results[1], img_show[0]);
	hconcat(results[2], results[3], img_show[1]);
	hconcat(img_show[0], img_show[1], show);

	string savename = "Canny_Result_" + to_string(index+1);
	imshow(savename, show);
	//imwrite("D:/ImgSeg/experiments/" + savename + ".png", img_show[0]);
	int key = waitKey(0);
	if (key == 27)
		destroyAllWindows();

	return EXIT_SUCCESS;
}

/* Laplacian of Gauss */
void LoG(Mat input, Size size, float sigma, int border_type, Mat& output) {
	Mat img_blurred = gaussBlur(input, size, sigma, border_type);
	Mat laplacian_filter = (Mat_<float>(3, 3) << 0.f, 1.f, 0.f, 1.f, -4.f, 1.f, 0.f, 1.f, 0.f);
	Mat edge;
	conv2D(img_blurred, laplacian_filter, edge, border_type);
	// convert the result to uint8
	edge.convertTo(output, CV_8UC3);
}

/* Prewitt operator */
void prewitt(const Mat& input, Mat& output, int x, int y, int borderType) {
	CV_Assert(!(x == 0 && y == 0));
	// horizontal gradient
	if (x != 0 && y == 0) {
		// seperable filter and conv.
		Mat prewitt_x_y = (Mat_<float>(3, 1) << 1, 1, 1);
		Mat prewitt_x_x = (Mat_<float>(1, 3) << 1, 0, -1);
		conv2D(input, prewitt_x_y, output, borderType);
		conv2D(output, prewitt_x_x, output, borderType);
	}
	// vertical gradient
	if (x == 0 && y != 0) {
		Mat prewitt_y_x = (Mat_<float>(1, 3) << 1, 1, 1);
		Mat prewitt_y_y = (Mat_<float>(3, 1) << 1, 0, -1);
		conv2D(input, prewitt_y_x, output, borderType);
		conv2D(output, prewitt_y_y, output, borderType);
	}
	output.convertTo(output, CV_8UC3);
}

/* Scharr operator */
void scharr(const Mat& input, Mat& output, int x, int y, int borderType) {
	CV_Assert(!(x == 0 && y == 0));
	Mat shcarr_x = (Mat_<float>(3, 3) << 3, 0, -3, 10, 0, -10, 3, 0, -3);
	Mat shcarr_y = (Mat_<float>(3, 3) << 3, 10, 3, 0, 0, 0, -3, -10, -3);
	// horizontal gradient
	if (x != 0 && y == 0) {
		conv2D(input, shcarr_x, output, borderType);
	}
	// vertical graddient
	if (x == 0 && y != 0) {
		conv2D(input, shcarr_y, output, borderType);
	}
	output.convertTo(output, CV_8UC3);
}

/* Sobel operator */
int factorial(int n) {
	int fac = 1;
	if (n == 0)
		return fac;
	for (int i = 1; i <= n; i++) {
		fac *= i;
	}
	return fac;
}

Mat getPascalSmooth(int n) {
	Mat pascalSmooth = Mat::zeros(Size(n, 1), CV_32FC1);
	for (int i = 0; i < n; i++) {
		pascalSmooth.at<float>(0, i) = factorial(n - 1) / (factorial(i) * factorial(n - 1 - i));
	}
	return pascalSmooth;
}

Mat getPascalDiff(int n) {
	Mat pascalDiff = Mat::zeros(Size(n, 1), CV_32FC1);
	Mat pascalSmooth_prev = getPascalSmooth(n - 1);
	for (int i = 0; i < n; i++) {
		if (i == 0)
			pascalDiff.at<float>(0, i) = 1;
		else if (i == n - 1)
			pascalDiff.at<float>(0, i) = -1;
		else
			pascalDiff.at<float>(0, i) = pascalSmooth_prev.at<float>(0, i) - 
									     pascalSmooth_prev.at<float>(0, i - 1);
	}
	return pascalDiff;
}

Mat sobel(Mat input, int dx, int dy, int winSize, int borderType) {
	CV_Assert(winSize >= 3 && winSize % 2 == 1);
	Mat pascalSmooth = getPascalSmooth(winSize);
	Mat pascalDiff = getPascalDiff(winSize);
	Mat result;

	// horizontal gradients -> get verticcal edges
	if (dx != 0) {
		// seperable conv, 1-D vertical conv. and 1-D horizontal conv.
		conv2D(input, pascalSmooth.t(), result, borderType);
		conv2D(result, pascalDiff, result, borderType);
	}
	// vetrical gradients -> get horizontal edges
	if (dx == 0 && dy != 0) {
		// seperable conv, 1-D horizontal conv. and then 1-D vertical conv.
		conv2D(input, pascalSmooth, result, borderType);
		conv2D(result, pascalDiff.t(), result, borderType);
	}
	//result.convertTo(result, CV_8UC3);
	return result;
}

/* Caculate the amplitude of gradient */
Mat calcMagnitude(Mat g_x, Mat g_y, norm_type type) {
	Mat amplitude;
	// L1-norm
	if (type == norm_type::L1) {
		amplitude = cv::abs(g_x) + cv::abs(g_x);
	}
	// L2-norm
	else if (type == norm_type::L2) {
		Mat g_x2, g_y2;
		g_x.convertTo(g_x, CV_32F);
		g_y.convertTo(g_y, CV_32F);
		cv::pow(g_x, 2.0, g_x2);
		cv::pow(g_y, 2.0, g_y2);
		cv::sqrt(g_x2 + g_y2, amplitude);
	}
	//amplitude.convertTo(amplitude, CV_8UC3);
	return amplitude;
}

/* Canny Detector */
Mat nonMaxSuppression(Mat dx, Mat dy, norm_type type) {
	Mat magnitude = calcMagnitude(dx, dy, type);
	const int rows = dx.rows;
	const int cols = dy.cols;
	Mat nonMaxSup = Mat::zeros(dx.size(), dx.type());
	for (int r = 1; r < rows - 1; r++) {
		for (int c = 1; c < cols - 1; c++) {
			float x = dx.at<float>(r, c);
			float y = dy.at<float>(r, c);
			// gradient direction
			float angle = (atan2f(y, x) / CV_PI) * 180;
			float mag = magnitude.at<float>(r, c);
			if (abs(angle) < 22.5 || abs(angle) > 157.5) {
				float left = magnitude.at<float>(r, c - 1);
				float right = magnitude.at<float>(r, c + 1);
				if (mag > left || mag > right)
					nonMaxSup.at<float>(r, c) = mag;
			}
			if ((angle >= 22.5 && angle < 67.5) || (angle < -112.5 && angle >= 157.5)) {
				float leftTop = magnitude.at<float>(r - 1, c - 1);
				float rightBottom = magnitude.at<float>(r + 1, c + 1);
				if (mag > leftTop || mag > rightBottom)
					nonMaxSup.at<float>(r, c) = mag;
			}
			if ((angle >= 67.5 && angle <= 112.5) || (angle >= -112.5 && angle <= -67.5)) {
				float top = magnitude.at<float>(r - 1, c);
				float bottom = magnitude.at<float>(r + 1, c);
				if (mag > top || mag > bottom)
					nonMaxSup.at<float>(r, c) = mag;
			}
			if ((angle > 112.5 && angle < 157.5) || (angle > -67.5 && angle <= -22.5)) {
				float rightTop = magnitude.at<float>(r + 1, c - 1);
				float leftBottom = magnitude.at<float>(r - 1, c + 1);
				if (mag > rightTop || mag > leftBottom)
					nonMaxSup.at<float>(r, c) = mag;
			}
		}
	}
	return nonMaxSup;
}

bool isInRange(int r, int c, int rows, int cols) {
	if (r >= 0 && r < rows && c >= 0 && c < cols)
		return true;
	return false;
}

void track(Mat nonMaxSup, Mat& edge, float lowerThresh, int r, int c, int rows, int cols) {
	if (edge.at<uchar>(r, c) == 0) {
		edge.at<uchar>(r, c) == 255;
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				float mag = nonMaxSup.at<float>(r + i, c + j);
				if (isInRange(r + i, c + j, rows, cols) && mag >= lowerThresh) {
					cout << "Track again." << endl;
					track(nonMaxSup, edge, lowerThresh, r + i, c + j, rows, cols);
				}
			}
		}
	}
}

Mat doubleThreshold(Mat nonMaxSup, float lowerThresh, float upperThresh) {
	const int rows = nonMaxSup.rows;
	const int cols = nonMaxSup.cols;
	Mat edge = Mat::zeros(Size(cols, rows), CV_8UC1);
	// double threshold
	for (int r = 1; r < rows - 1; r++) {
		for (int c = 1; c < cols - 1; c++) {
			float mag = nonMaxSup.at<float>(r, c);
			if (mag >= upperThresh)
				edge.at<uchar>(r, c) = 255;
			if (mag < lowerThresh)
				edge.at<uchar>(r, c) = 0;
		}
	}
	Mat finalEdge;
	edge.copyTo(finalEdge);	
	// hysteresis	
	for (int r = 1; r < rows - 1; r++) {
		for (int c = 1; c < cols - 1; c++) {
			float mag = nonMaxSup.at<float>(r, c);
			if (mag >= lowerThresh && mag < upperThresh) {
				if (edge.at<uchar>(r - 1, c - 1) == 255 or edge.at<uchar>(r, c - 1) == 255 or
					edge.at<uchar>(r + 1, c - 1) == 255 or edge.at<uchar>(r - 1, c) == 255 or
					edge.at<uchar>(r + 1, c) == 255 or edge.at<uchar>(r - 1, c + 1) == 255 or
					edge.at<uchar>(r, c + 1) == 255 or edge.at<uchar>(r + 1, c + 1) == 255)
					finalEdge.at<uchar>(r, c) = 255;
				else
					finalEdge.at<uchar>(r, c) = 0;
			}
			
		}
	}	
	return finalEdge;
}

void CannyDetect(Mat input, Mat& output, float lowerThresh, float upperThresh, norm_type type) {
	Mat gradient_x, gradient_y;
	gradient_x = sobel(input, 1, 0, 3, BORDER_REFLECT);
	gradient_y = sobel(input, 0, 1, 3, BORDER_REFLECT);
	cvtColor(gradient_x, gradient_x, COLOR_BGR2GRAY);
	cvtColor(gradient_y, gradient_y, COLOR_BGR2GRAY);
	Mat nonMaxSup = nonMaxSuppression(gradient_x, gradient_y, type);
	output = doubleThreshold(nonMaxSup, lowerThresh, upperThresh);
}
void conv2D(Mat input, Mat kernel, Mat& output, int borderType)
{
	CV_Assert(input.channels() == 3);
	const int rows = input.rows;
	const int cols = input.cols;
	const int kernel_h = kernel.rows;
	const int kernel_w = kernel.cols;
	const int h = (kernel_h - 1) / 2;
	const int w = (kernel_w - 1) / 2;

	Mat region;
	vector<Mat> input_chs;
	split(input, input_chs);
	for (Mat& m : input_chs) {
		copyMakeBorder(m, m, h, h, w, w, borderType);
	}

	Mat b(input.size(), CV_32FC1), g(input.size(), CV_32FC1), r(input.size(), CV_32FC1);
	vector<Mat> output_chs{ b, g, r };

	for (int ch = 0; ch < 3; ch++) {
		for (int r = h; r < h + rows; r++) {
			for (int c = w; c < w + cols; c++) {
				input_chs[ch](Rect(c - w, r - h, kernel_w, kernel_h)).convertTo(region, CV_32FC1);
				output_chs[ch].at<float>(r - h, c - w) = (float)region.dot(kernel);
			}
		}
	}
	merge(output_chs, output);
}

Mat gaussBlur(Mat input, Size ksize, float sigma, int borderType)
{
	CV_Assert(input.channels() == 3);
	CV_Assert(ksize.width % 2 == 1 && ksize.height % 2 == 1);
	// construct gaussian kernel in y and x direction
	Mat gaussKernel_y = getGaussianKernel(ksize.height, sigma, CV_32F);
	Mat gaussKernel_x = getGaussianKernel(ksize.width, sigma, CV_32F);
	gaussKernel_x = gaussKernel_x.t();
	// separable gaussian convolution
	Mat output(input.size(), CV_32FC3), conv_y(input.size(), CV_32FC3);
	conv2D(input, gaussKernel_y, conv_y, BORDER_REPLICATE);
	conv2D(conv_y, gaussKernel_x, output, BORDER_REPLICATE);
	output.convertTo(output, CV_8UC3);

	return output;
}
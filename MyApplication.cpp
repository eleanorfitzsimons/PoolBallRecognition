// myApplication.cpp

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>


using namespace cv;
using namespace std;



//////////////////////////////////
// STAGE 1: Dectecting the balls
//////////////////////////////////


//perspective transform
Mat perspectiveTransform(const Mat& original) {

	//define source and destination points
	Point2f source_points[4] = {
		Point2f(337.0f, 35.0f),
		Point2f(1570.0f, 21.0f),
		Point2f(38.0f, 472.0f),
		Point2f(1888.0f, 451.0f)
	};
	Point2f destination_points[4] = {
		Point2f(0.0f, 0.0f),
		Point2f(1433.0f, 0.0f),
		Point2f(0.0f, 682.0f),
		Point2f(1433.0f, 682.0f)
	};

	//calculate taransform matrix
	Mat perspective_matrix = getPerspectiveTransform(source_points, destination_points);

	//apply transform
	Mat transformed;
	warpPerspective(original, transformed, perspective_matrix, Size(1434, 683));

	//return transformed image
	return transformed;
}

/*
//segment table... seprate green from balls
Mat segmentTable(Mat transformed) {}


//find positions of balls
vector<Point2f> findBalls(Mat mask) {} //returns vector of positions of balls in the image 
	//look over vectors


////////////////////////////////////
// STAGE 2: Recognising the balls
////////////////////////////////////


int recogniseBalls(Mat transformed, vector<Point2f> ballPositions) {}



///////////////////////////////////
// EVAL
//////////////////////////////////

//predicted positions vs gt
void evalDetection() {}
	

//predicted balls vs gt
void evalRecognition() {}




*/



void MyApplication() {
	
	//load images
	Mat original = imread("Table1.jpeg"); //ok fix this here for all

	if(original.empty()) {
		cout << "Could not open or find the image" << endl;
		return;

	}

	//perspective transform
	Mat transformed = perspectiveTransform(original);
	


	/*
	//segment background
	Mat mask = segmentTable(transformed);

	//detect balls
	vector<Point2f> ballPositions = findBalls(mask);
	//reconginse balls
	
	//evaluate results
	*/
	imshow("Original Image", original);
	imshow("Transformed Image", transformed);
	waitKey(0); //keep window open  
}

int main() {
	MyApplication();
	return 0;
}
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
Mat perspectiveTransform(Mat original) {

	//define source and destination points
	source_points[4] = {
		point2f(337.0f, 35.0f),
		point2f(1570.0f, 21.0f),
		point2f(38.0f, 4720.0f),
		point2f(1888.0f, 451.0f)
	};
	destination_points[4] = {
		point2f(0.0f, 0.0f),
		point2f(1433.0f, 0.0f),
		point2f(0.0f, 682.0f),
		point2f(1433.0f, 682.0f)
	};

	//calculate taransform matrix
	perspective_matrix = getPerspectiveTransform(source_points, destination_points);

	//apply transform
	Mat transformed;
	warpPerspective(original, transformed, perspective_matrix, Size(1434, 683));

	//return transformed image
	return trandformed;
}

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








void MyApplication() {

	//load images
	Mat original = imread("Table1.jpeg"); //ok fix this here for all

	//perspective transform
	Mat transformed = perspectiveTransform(original);

	//segment background
	Mat mask = segmentTable(transformed);

	//detect balls
	vector<Point2f> ballPositions = findBalls(mask);
	//reconginse balls
	
	//evaluate results
}
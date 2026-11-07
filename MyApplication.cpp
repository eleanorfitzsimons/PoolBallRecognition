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
Mat perspectiveTransform(Mat original) {}

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
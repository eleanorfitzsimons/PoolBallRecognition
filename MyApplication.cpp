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




//find potential circles
vector<vec3f> findBalls(const Mat& original) {
	//changed to 3f now we keep the circumference
	

	//converting to gs 
	Mat gray;
	cvtColor(original, gray, COLOR_BGR2GRAY);

	//reduce noise
	Mat blurred;
	GaussianBlur(gray, blurred, Size(5, 5), 1.5); //ok tune parameters?????


	//detect circles using HoughCircles
	vector<Vec3f> circles;

	HoughCircles(
		blurred,
		circles,
		HOUGH_GRADIENT,
		1, //dp accymulator resolution
		30, //min distance between circles
		100, //param1: higher threshold for Canny edge detector  
		10, //param2: threshold for center detection
		15, //min radius
		28 //max radius
	);
	
	return circles;
} //returns vector of positions of balls in the image 

//filter balls, remove fasle pos
vector<Point2f> filterBalls(const vector<Point2f>& possible) {

	vector<Point2f> validBalls;

	//within table coords
	vector<Point> tableCorners = {
		Point(337, 35),
		Point(1570, 21),
		Point(1888, 451),
		Point(38, 472)
	};
	
	for (const Point2f& centre : possible) {

		double inside = pointPolygonTest(
			tableCorners,
			centre,
			false
		);

		if (inside >= 0) {
			validBalls.push_back(centre);
		}
	}
	//colour 
	//could use old mask
	//we akso know has to be wi table corners
	return validBalls;
}

vector<Point2f> transformBallPosititions(const vector<Point2f>& ogPositions) {
	//perspective transform
}

Mat segmentTable(){
	//estimate the cloth colout and create fg mask
	//go back and get the old fn
}
/*
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

	//detect balls

	vector<Point2f> potential = findBalls(original);
	vector<Point2f> ballPositions = filterBalls (potential);
	Mat filtered = original.clone();
	for (int i = 0; i < ballPositions.size(); i++) {
		circle(filtered, ballPositions[i], 5, Scalar(0, 0, 255), FILLED);
	}
	imshow("filtered result", filtered);

	/*
	
	//detect balls
	vector<Point2f> ballPositions = findBalls(mask);
	//reconginse balls
	
	//evaluate results
	*/
	//imshow("Original Image", original);
	//imshow("Transformed Image", transformed);
	waitKey(0); //keep window open  
}

int main() {
	MyApplication();
	return 0;
}

//the transformation not being perfect, criticise gt

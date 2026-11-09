// myApplication.cpp

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>


using namespace cv;
using namespace std;


//////////////////////////////////
// STAGE 1: Dectecting the balls 
//////////////////////////////////

//find potential circles
vector<Vec3f> findBalls(const Mat& original) {
	//changed to 3f now we keep the radius
	

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
		8, //param2: threshold for center detection
		15, //min radius
		28 //max radius
	);
	
	return circles;
} //returns vector of positions of balls in the image 
//filter balls, remove fasle pos
vector<Vec3f> filterBalls(const vector<Vec3f>& possible, const Mat& fgMask) {

    vector<Vec3f> validBalls;

    //FILTER INSIDE TABLE
    vector<Point> tableCorners = {
        Point(337, 35),
        Point(1570, 21),
        Point(1888, 451),
        Point(38, 472)
    };

    for (const Vec3f& ball : possible) {

        Point2f centre(ball[0], ball[1]);
        int radius = cvRound(ball[2]);

        //from centre to table boundary
        double distance = pointPolygonTest(
            tableCorners,
            centre,
            true
        );

        //ignore balls too close to table edges
        if (distance < 0.75 * radius) {
            continue;
        }

        //FILTER WITH MASK

        //mask for filled circle
        Mat circleMask = Mat::zeros(fgMask.size(), CV_8UC1);

        circle(
            circleMask,
            centre,
            max(1, radius - 3),
            Scalar(255),
            FILLED
        );

        int totalPixels = countNonZero(circleMask);

        Mat foregroundInside;
        bitwise_and(fgMask, circleMask, foregroundInside); //combine two masks and keep only if white in both

        int foregroundPixels = countNonZero(foregroundInside);

        float foregroundRatio =
            static_cast<float>(foregroundPixels) / totalPixels;

        if (foregroundRatio > 0.5f) { //TUNE
            validBalls.push_back(ball);
        }
    }

    return validBalls;
}

vector<Point2f> transformBallPosititions(const vector<Point2f>& ogPositions) {
    //perspective transform

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

    //calculate transform matrix
    Mat perspective_matrix =
        getPerspectiveTransform(source_points, destination_points);

    //apply transform to ball positions
    vector<Point2f> transformedPositions;

    if (!ogPositions.empty()) {
        cv::perspectiveTransform(
            ogPositions,
            transformedPositions,
            perspective_matrix
        );
    }

    return transformedPositions;
}

//adding correct ball position funciton to account for the fact the balls dont like ont he palnne
//or could just justify paralax errors in report.. not knowign camera hiehgt


int findBgHue(const Mat& hsv){
    //calculate hue histogram
    //2d hue saturation for robustness to lighting??????

    //histogram parameters
    int histSize = 30;                            //bins (each being 6)
    float hueRange[] = {0, 180};                  //values covered by hist
    const float* ranges[] = {hueRange};           //format hist fn needs
    int channels[] = {0};

    //calculate histogram
    Mat hueHist;
    calcHist(
        &hsv,       //input
        1,          //no of images
        channels,   //channel - 0 is hue
        Mat(),      //no mask, using all px ????????????? could change???
        hueHist,    //output
        1,          //dimensions
        &histSize,  //num bins
        ranges      //hue range
    );

    //find most frequent bin
    Point maxLocation;
    minMaxLoc(
        hueHist,
        nullptr,
        nullptr,
        nullptr,
        &maxLocation
    );

    int dominantBin = maxLocation.y;

    //bin index to hue val
    float binWidth = 180.0f / histSize;

    int bgHue = static_cast<int>((dominantBin + 0.5f) * binWidth);

    return bgHue;
}

Mat segmentTable(const Mat& original) {

    //mean shift
    Mat filtered;
    pyrMeanShiftFiltering(original, filtered, 10, 20, 1);

    //hsv
    Mat hsv;
    cvtColor(filtered, hsv, COLOR_BGR2HSV);

    //bg hue fn (uses hist)
    int bgHue = findBgHue(hsv);

    //find average saturation of cloth-like pixels
    const int HUE_TOLERANCE = 12; //TUNE

    Mat hueMask = Mat::zeros(hsv.size(), CV_8UC1);

    for (int y = 0; y < hsv.rows; y++) {
        for (int x = 0; x < hsv.cols; x++) {

            int hue = hsv.at<Vec3b>(y, x)[0];

            int difference = abs(hue - bgHue);
            difference = min(difference, 180 - difference);

            if (difference <= HUE_TOLERANCE) {
                hueMask.at<uchar>(y, x) = 255;
            }
        }
    }

    Mat channels[3];
    split(hsv, channels);

    double bgSat = mean(channels[1], hueMask)[0];

    //binary bg mask
    const int SAT_TOLERANCE = 30; //TUNE
    Mat bgMask = Mat::zeros(hsv.size(), CV_8UC1);

    for (int y = 0; y < hsv.rows; y++) {
        for (int x = 0; x < hsv.cols; x++) {

            Vec3b pixel = hsv.at<Vec3b>(y, x);

            int hue = pixel[0];
            int saturation = pixel[1];

            int difference = abs(hue - bgHue);
            difference = min(difference, 180 - difference);

            //classify as cloth if hue AND saturation are similar
            if (difference <= HUE_TOLERANCE &&
                abs(saturation - bgSat) <= SAT_TOLERANCE) {

                bgMask.at<uchar>(y, x) = 255;
            }
        }
    }

    //invert
    Mat fgMask;
    bitwise_not(bgMask, fgMask);

    return fgMask;
}
/*
////////////////////////////////////
// STAGE 2: Recognising the balls
////////////////////////////////////


int recogniseBalls(Mat transformed, vector<Point2f> ballPositions) {

}
//


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
	Mat original = imread("Pool/Table1.jpeg"); //ok fix this here for all

	if(original.empty()) {
		cout << "Could not open or find the image" << endl;
		return;

	}

	//detect balls
	vector<Vec3f> potential = findBalls(original);

	//segment table
	Mat fgMask = segmentTable(original);

	//filter potential balls
	vector<Vec3f> ballPositions = filterBalls(potential, fgMask);

    //get ball centres
    vector<Point2f> ogPositions;

    for (const Vec3f& ball : ballPositions) {
        ogPositions.push_back(Point2f(ball[0], ball[1]));
    }

    //transform ball positions to table coords (x,y)
    vector<Point2f> tablePositions = transformBallPosititions(ogPositions);
    //transform image for checking ball positions
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

    Mat perspective_matrix =
        getPerspectiveTransform(source_points, destination_points);

    Mat transformed;
    warpPerspective(original, transformed, perspective_matrix, Size(1434, 683));

    //draw transformed ball centres
    for (const Point2f& centre : tablePositions) {
        circle(transformed, centre, 5, Scalar(0, 0, 255), FILLED);
    }

    imwrite("transformed_result.png", transformed);
    //print and compare to ground truth
    cout << "\ntransformed:\n";
    cout << "Ball\tRow\tColumn\n";

    for (int i = 0; i < tablePositions.size(); i++){
        cout << i +1 << "\t\t"
            << cvRound(tablePositions[i].y) << "\t"
            << cvRound(tablePositions[i].x) <<endl;
    }

    
    //check raw Hough detections
    Mat houghResult = original.clone();

    for (const Vec3f& ball : potential) {

        Point centre(cvRound(ball[0]), cvRound(ball[1]));

        circle(houghResult, centre, 5, Scalar(0, 0, 255), FILLED);
    }

    imwrite("hough_result.png", houghResult);

    //draw detected ball positions
    Mat filtered = original.clone();

    for (const Vec3f& ball : ballPositions) {
        Point centre(cvRound(ball[0]), cvRound(ball[1]));
        circle(filtered, centre, 5, Scalar(0, 0, 255), FILLED);
    }

    //save results
    imwrite("filtered_result.png", filtered);
    imwrite("foreground_mask.png", fgMask);

    cout << "Hough candidates: " << potential.size() << endl;
    cout << "Filtered candidates: " << ballPositions.size() << endl;
        

	//reconginse balls
	
	//evaluate results
	
	//imshow("Original Image", original);
	//imshow("Transformed Image", transformed);
	//waitKey(0); //keep window open  
}

int main() {
	MyApplication();
	return 0;
}

//the transformation not being perfect, criticise gt
//for the mask nexct is to maybe reduce the area we are histogramming
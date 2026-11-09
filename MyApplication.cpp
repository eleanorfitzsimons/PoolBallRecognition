// myApplication.cpp

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

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

    //mask for pixels inside table polygon
    vector<Point> tableCorners = {
        Point(337, 35),
        Point(1570, 21),
        Point(1888, 451),
        Point(38, 472)
    };

    Mat tableMask = Mat::zeros(hsv.size(), CV_8UC1);
    fillConvexPoly(tableMask, tableCorners, Scalar(255));


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
        tableMask,      //no mask, using all px ????????????? could change???
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

////////////////////////////////////
// STAGE 2: Recognising the balls
////////////////////////////////////

//colour histogram from one ball
Mat extractBallHist(){

}
//train model
vector<Mat> trainBallRecognition(){};

//score with bakc proj
float scoreBallMatch(){};

int recogniseBalls(Mat transformed, vector<Point2f> ballPositions) {
    
    //train
        //take all balls training image

        //perspective transform

        //use lableled ball pos to build histogram

    //rec
        //detect balls and transform images

        //back project reference hsitograms onto detected ballregions

        //predict ball ientiy

}




///////////////////////////////////
// EVAL
//////////////////////////////////

//load ground truth from csv file
vector<Point2f> loadGroundTruth(const string& imageName) {

    vector<Point2f> groundTruth;

    //open csv file
    ifstream file("Ball Locations In Transformed Image (Second correction).csv");

    if (!file.is_open()) {
        cout << "Could not open ground truth file" << endl;
        return groundTruth;
    }

    //skip two header rows
    string line;
    getline(file, line);
    getline(file, line);
    //find row for the selected image
    while (getline(file, line)) {

        stringstream ss(line);
        string imageNameCSV;

        //read image filename
        getline(ss, imageNameCSV, ',');

        if (imageNameCSV != imageName) {
            continue;
        }

        //read ball positions (1-15 and white)
        for (int i = 0; i < 16; i++) {

            string row, column;

            getline(ss, row, ',');
            getline(ss, column, ',');

            //skip balls with missing positions
            if (row.empty() || column.empty()) {
                continue;
            }

            //OpenCV uses (x,y) = (column,row)
            groundTruth.push_back(
                Point2f(stof(column), stof(row))
            );
        }

        break; //found the correct image
    }

    return groundTruth;
}

//predicted positions vs gt
void evalDetection(const vector<Point2f> & tablePositions, const vector<Point2f> & groundTruth) {
    //max distance for correct match
    const float MATCH_TOLERANCE = 15; ////TUNEEE

    //counters
    int TP = 0;
    int FP = 0;
    int FN = 0;

    //tracking whihc gt balls are matched
    vector<bool> matched(groundTruth.size(), false);

    vector<tuple<float, int, int>> possibleMatches;
    //loop through predicted positions
    for (int i = 0; i < tablePositions.size(); i++) {

        //compare with every ground truth position
        for (int j = 0; j < groundTruth.size(); j++) {

            //calculate Euclidean distance
            float distance = norm(tablePositions[i] - groundTruth[j]);

            //only consider positions within tolerance
            if (distance <= MATCH_TOLERANCE) {
                possibleMatches.push_back(make_tuple(distance, i, j));
            }
        }
    }


    //unmatched balls as fn
    //sort possible matches by distance (smallest first)
    sort(possibleMatches.begin(), possibleMatches.end());

    //track which predicted balls have been matched
    vector<bool> predictedMatched(tablePositions.size(), false);

    //record position errors
    vector<float> positionErrors;

    //match closest pairs
    for (const auto& match : possibleMatches) {

        float distance = get<0>(match);
        int i = get<1>(match); //predicted ball index
        int j = get<2>(match); //ground truth ball index

        //only match if neither ball has already been used
        if (!predictedMatched[i] && !matched[j]) {

            predictedMatched[i] = true;
            matched[j] = true;

            TP++;
            positionErrors.push_back(distance);
        }
    }
    FP = static_cast<int>(tablePositions.size()) - TP;
    FN = static_cast<int>(groundTruth.size()) - TP;

    //metrics
    float precision = 0.0f;
    float recall = 0.0f;
    float F1 = 0.0f;

    if (TP + FP > 0) {
        precision = static_cast<float>(TP) / (TP + FP);
    }

    if (TP + FN > 0) {
        recall = static_cast<float>(TP) / (TP + FN);
    }

    if (precision + recall > 0) {
        F1 = 2 * precision * recall / (precision + recall);
    }

    //mean position error
    float meanError = 0.0f;

    for (float error : positionErrors) {
        meanError += error;
    }

    if (!positionErrors.empty()) {
        meanError /= positionErrors.size();
    }

    //print results
    cout << "\nDETECTION EVALUATION\n";
    cout << "TP: " << TP << endl;
    cout << "FP: " << FP << endl;
    cout << "FN: " << FN << endl;
    cout << "Precision: " << precision << endl;
    cout << "Recall: " << recall << endl;
    cout << "F1: " << F1 << endl;
    cout << "Mean position error: " << meanError << " pixels" << endl;

}
	

//predicted balls vs gt
//void evalRecognition() {}








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
    //load ground truth positions from csv
    vector<Point2f> groundTruth = loadGroundTruth("Table1.jpeg");

    //evaluate detection
    if (!groundTruth.empty()) {
        evalDetection(tablePositions, groundTruth);
    }
    else {
        cout << "Skipping evaluation: no ground truth loaded" << endl;
    }
	
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
//mask needs to be changed for saturation

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
// TABLE GEOMETRY 
//////////////////////////////////

//table corners in og image, order: top left, top right, bottom right, bottom left

const vector<Point2f> TABLE_CORNERS = {
    Point2f(337.0f, 35.0f),
    Point2f(1570.0f, 21.0f),
    Point2f(1888.0f, 451.0f),
    Point2f(38.0f, 472.0f)
};

//size of the transf table image
const float TABLE_W = 1433.0f;
const float TABLE_H = 682.0f;
const Size WARP_SIZE(1434, 683);

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

    for (const Vec3f& ball : possible) {

        Point2f centre(ball[0], ball[1]);
        int radius = cvRound(ball[2]);

        //from centre to table boundary
        double distance = pointPolygonTest(
            TABLE_CORNERS,
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


//calculate perspective transformation matrix
Mat getTablePerspectiveMatrix() {

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

    return getPerspectiveTransform(source_points, destination_points);
}

vector<Point2f> transformBallPosititions(const vector<Point2f>& ogPositions) {
    //perspective transform


    //calculate transform matrix
    Mat perspective_matrix = getTablePerspectiveMatrix();

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
   vector<Point> tablePoly(TABLE_CORNERS.begin(), TABLE_CORNERS.end());
    Mat tableMask = Mat::zeros(hsv.size(), CV_8UC1);
    fillConvexPoly(tableMask, tablePoly, Scalar(255));


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
vector<Point2f> loadGroundTruth(const string& imageName);//forward declaration, these might need to be reordered


//colour histogram from one ball
Mat extractBallHist(const Mat& hsv, Point2f centre, Size axes){
    //circuilar mask for ball
    Mat ballMask = Mat::zeros(hsv.size(), CV_8UC1);


    //elliptical mask for transformed ball
    ellipse(
        ballMask,
        centre,
        axes,
        0,
        0,
        360,
        Scalar(255),
        FILLED
    );

    //histogram parameters
    int histSize[] = {30, 32}; //hue and saturation bins

    float hueRange[] = {0, 180};
    float satRange[] = {0, 256};

    const float* ranges[] = {hueRange, satRange};
    int channels[] = {0, 1}; //hue and saturation

    //calculate HSV histogram
    Mat hist;
        calcHist(
        &hsv,
        1,
        channels,
        ballMask,
        hist,
        2,
        histSize,
        ranges
    );
        //normalise histogram
    normalize(hist, hist, 1.0, 0.0, NORM_L1);

    return hist;

}
//train model
vector<Mat> trainBallRecognition(){
    //load the training image
    Mat trainingImage = imread("Pool/AllBalls.jpeg");

    if (trainingImage.empty()) {
        cout << "Could not load training image" << endl;
        return {};
    }

    //load labelled positions from csv
    vector<Point2f> referencePositions = loadGroundTruth("AllBalls.jpeg");

    //perspective transform on training image
    //.......
    Mat perspective_matrix = getTablePerspectiveMatrix();

    Mat transformed;
    warpPerspective(trainingImage, transformed, perspective_matrix, Size(1434,683));
    //check labelled training ball positions
    Mat trainingResult = transformed.clone();

    for (int i = 0; i < referencePositions.size(); i++) {

        //draw sampling area around each training ball
        ellipse(
            trainingResult,
            referencePositions[i],
            Size(16, 30),
            0,
            0,
            360,
            Scalar(0, 0, 255),
            1
        );

        //label reference ball
        string label = (i == 15) ? "W" : to_string(i + 1);

        putText(
            trainingResult,
            label,
            referencePositions[i] + Point2f(-8, -18),
            FONT_HERSHEY_SIMPLEX,
            0.4,
            Scalar(255, 255, 255),
            1
        );
    }

    imwrite("training_samples.png", trainingResult);
    //convert to hsv
    Mat hsv;
    cvtColor(transformed, hsv, COLOR_BGR2HSV);

    //histogram for each labeled ball
    vector <Mat> referenceHistograms;
    const Size BALL_AXES(16, 30); //TUNE
    for (const Point2f & centre : referencePositions){
        Mat hist = extractBallHist(hsv, centre, BALL_AXES);
        referenceHistograms.push_back(hist);
    }

    return referenceHistograms;

};

//score with bakc proj
float scoreBallMatch(const Mat& hsv, Point2f centre, Size axes, const Mat& referenceHist){

    //histogram parameters - same channels and ranges as training
    int channels[] = {0, 1};

    float hueRange[] = {0, 180};
    float satRange[] = {0, 256};

    const float* ranges[] = {hueRange, satRange};

    //convert HSV to float to preserve small back projection scores
    Mat hsvFloat;
    hsv.convertTo(hsvFloat, CV_32F);

    //back project reference histogram onto test image
    Mat backProjection;

    calcBackProject(
        &hsvFloat,
        1,
        channels,
        referenceHist,
        backProjection,
        ranges,
        1.0,
        true
    );

    //create circular mask around detected ball
    Mat ballMask = Mat::zeros(hsv.size(), CV_8UC1);

    //elliptical mask around detected ball
    ellipse(
        ballMask,
        centre,
        axes,
        0,
        0,
        360,
        Scalar(255),
        FILLED
    );

    //average back-projection response inside ball
    float score = static_cast<float>(
        mean(backProjection, ballMask)[0]
    );

    return score;
};

vector<int> recogniseBalls(const Mat & transformed, const vector<Point2f> &ballPositions, const vector<Mat> & referenceHistogram ) {
    

    vector<int> predictedBalls;

    //convert transformed image to HSV
    Mat hsv;
    cvtColor(transformed, hsv, COLOR_BGR2HSV);

    const Size BALL_AXES(16, 30);


    //recognise each detected ball
    for (const Point2f& centre : ballPositions) {

        float bestScore = -1.0f;
        int bestBall = -1;

        //ball type from px inside elipse
        Mat mask = Mat::zeros(hsv.size(), CV_8UC1);
        ellipse(mask, centre, BALL_AXES, 0, 0, 360, Scalar(255), FILLED);

        Mat whitePix, darkPix;
        inRange(hsv, Scalar(0, 0, 150), Scalar(180, 60, 255), whitePix); //bright, low saturation
        inRange(hsv, Scalar(0, 0, 0), Scalar(180, 255, 60), darkPix);    //very low brightness

        //fraction of white px vs dark... strip or solid
        float total = static_cast<float>(countNonZero(mask));
        float whiteFrac = countNonZero(whitePix & mask) / total;
        float darkFrac = countNonZero(darkPix & mask) / total;

                //range of reference indices this ball is allowed to match (0-6 = balls 1-7, 7 = ball 8, 8-14 = 9-15, 15 = white)
        int lo, hi;
        if (whiteFrac > 0.75f)      { lo = 15; hi = 15; } //white ball
        else if (darkFrac > 0.5f)   { lo = 7;  hi = 7;  } //8 ball
        else if (whiteFrac > 0.25f) { lo = 8;  hi = 14; } //stripe
        else                        { lo = 0;  hi = 6;  } //solid


        cout << "whiteFrac " << whiteFrac << "  darkFrac " << darkFrac << endl; //for checking thresholds

        //compare against allowed reference histograms
        for (int i = lo; i <= hi && i < referenceHistogram.size(); i++) {
            if (referenceHistogram[i].empty()) continue;
            float score = scoreBallMatch(
                hsv,
                centre,
                BALL_AXES,
                referenceHistogram[i]
            );

            if (score > bestScore) {
                bestScore = score;

                //reference order: balls 1-15, then white
                bestBall = (i == 15) ? 0 : i + 1;
            }
        }

        predictedBalls.push_back(bestBall);
    }

    return predictedBalls;
}





////////////////////////////////////////
// EVAL
////////////////////////////////////////

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
vector<int> evalDetection(const vector<Point2f> & tablePositions, const vector<Point2f> & groundTruth) {
    //max distance for correct match
    const float MATCH_TOLERANCE = 15; ////TUNEEE

    //counters
    int TP = 0;
    int FP = 0;
    int FN = 0;

    //tracking whihc gt balls are matched
    vector<bool> matched(groundTruth.size(), false);
    //true ball identity for each predicted position: 0 = white, 1-15 = numbered, -1 = false positive
    vector<int> trueLabels(tablePositions.size(), -1);

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

            trueLabels[i] = (j == 15) ? 0 : j + 1;

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
    
    return trueLabels;
}
	

//predicted balls vs gt
//index 0 = white ball, 1-15 = numbered balls
void evalRecognition(const vector<int>& predicted, const vector<int>& trueLabels) {
    Mat confusion = Mat::zeros(16, 16, CV_32S); //row = true ball, column = predicted ball
    int correct = 0;
    int total = 0;

    for (size_t i = 0; i < predicted.size(); i++) {

        //false positives have no true identity, and unrecognised balls can't be scored
        if (trueLabels[i] < 0 || predicted[i] < 0) continue;

        confusion.at<int>(trueLabels[i], predicted[i])++;
        total++;

        if (predicted[i] == trueLabels[i]) correct++;
    }
        cout << "\nRECOGNITION EVALUATION\n";
    cout << "Correct: " << correct << " / " << total << endl;

    if (total > 0) {
        cout << "Accuracy: " << static_cast<float>(correct) / total << endl;
    }

    //print the confusion matrix
    cout << "\nConfusion matrix (rows = true, columns = predicted, 0 = white)\n";

    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 16; c++) {
            cout << confusion.at<int>(r, c) << "\t";
        }
        cout << endl;
    }
}








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
    Mat perspective_matrix = getTablePerspectiveMatrix();

    Mat transformed;
    warpPerspective(original, transformed, perspective_matrix, Size(1434, 683));

 //make a copy for drawing so recognition uses the clean image
    Mat transformedResult = transformed.clone();

    //draw transformed ball centres
    for (const Point2f& centre : tablePositions) {
        circle(transformedResult, centre, 5, Scalar(0, 0, 255), FILLED);
    }

    imwrite("transformed_result.png", transformedResult);

    //print and compare to ground truth
    cout << "\ntransformed:\n";
    cout << "Candidate\tRow\tColumn\n";

    for (int i = 0; i < tablePositions.size(); i++) {
        cout << i + 1 << "\t\t"
             << cvRound(tablePositions[i].y) << "\t"
             << cvRound(tablePositions[i].x) << endl;
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

    //train using AllBalls.jpeg
    vector<Mat> referenceHistograms = trainBallRecognition();

    //recognise detected balls using back projection
    vector<int> predictedBalls =
        recogniseBalls(transformed, tablePositions, referenceHistograms);

    //print recognised ball identities
    cout << "\nRECOGNITION RESULTS\n";

    for (int i = 0; i < predictedBalls.size(); i++) {
        cout << "Candidate " << i + 1
             << ": Ball " << predictedBalls[i] << endl;
    }


    //evaluate results

    //load ground truth positions from csv
    vector<Point2f> groundTruth = loadGroundTruth("Table1.jpeg");

    //evaluate detection
    if (!groundTruth.empty()) {
        vector<int> trueLabels = evalDetection(tablePositions, groundTruth);
        evalRecognition(predictedBalls, trueLabels);
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


//striped vs solid
//change 
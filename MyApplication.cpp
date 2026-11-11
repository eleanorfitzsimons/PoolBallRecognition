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
		12, //param2: threshold for center detection
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
        if (distance < 0.6 * radius) {
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

//dark ball recovery
//second go at hough to find the ones we missed
vector<Vec3f> recoverDarkBalls(const Mat&original, const vector<Vec3f> & accepted, const Mat& fgMask){
    //keep the balls we already have           
    vector<Vec3f> result = accepted;

    //convert to gs for circles
    Mat gray, blurred, edges, hsv;
    cvtColor(original, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 1.5); //blur to reduce noise
    Canny(gray, edges, 70, 140); 
    cvtColor(original, hsv, COLOR_BGR2HSV); //for checking brightnes

    //second hough with lower threshold
    vector<Vec3f> weakCircles;
    HoughCircles(blurred, weakCircles, HOUGH_GRADIENT,
                 1, 30, 100, 7, 15, 28); //lower threshold 7 so more possible circles are found
        //go through the new possible circles
    for (const Vec3f& ball : weakCircles) {

        Point2f centre(ball[0], ball[1]);
        float radius = ball[2];

        //check if this ball has already been found
        bool duplicate = false;
        for (const Vec3f& oldBall : result){
            //compare distance between centres
            if(norm(centre - Point2f(oldBall[0], oldBall[1]))< 24.0f ){
                duplicate = true;
                break;
            }
        }
        //skip if ball is already on the list
        if(duplicate) 
            continue;
        if (pointPolygonTest(TABLE_CORNERS, centre, true) < 0.6f * radius)
            continue;

        //inside image
        const int R = cvRound(radius);
        const int cx = cvRound(centre.x);
        const int cy = cvRound(centre.y);
        const int extent = 2 * R;

        if (cx - extent < 0 || cy - extent < 0 || cx + extent >= original.cols || cy + extent >= original.rows)
            continue;

        //vals to measure brightness
        double innerBrightness = 0.0;
        double surroundBrightness = 0.0;

        int innerCount = 0;
        int surroundCount = 0;

        //count for edge px and fg px
        int edgeCount = 0; 
        int edgePixels = 0;
        int foregroundCount = 0;
        int diskCount = 0;

        //look at px aroudn ball
        for (int dy = -extent; dy <= extent; ++dy) {
            for (int dx = -extent; dx <= extent; ++dx) {

                //dist of px from centre
                float distance = sqrt(float(dx * dx + dy * dy)); //euclidian dist

                int x = cx + dx;
                int y = cy + dy;

                //brighness from v channlel in hsv
                int brightness = hsv.at<Vec3b>(y,x)[2];

                //brightness inside ball
                if (distance < 0.68f * radius){
                    innerBrightness += brightness;
                    ++innerCount;
                }

                //brightness of area round ball
                if (distance > 1.2f * radius && distance < 1.7f * radius){
                    surroundBrightness += brightness;
                    ++surroundCount;
                }

                //check edges near the circle boundary
                if (distance > 0.75f * radius && distance < 1.15f * radius) {

                    ++edgePixels;

                    if (edges.at<uchar>(y, x) > 0)
                        ++edgeCount;
                }

                //check how much of the ball is foreground
                if (distance <= max(1.0f, float(R - 3))) {

                    ++diskCount;

                    if (fgMask.at<uchar>(y, x) > 0)
                        ++foregroundCount;
                
                }
            }//end of dx loop
        }//end of dy loop

        //avoid zero div
        if(innerCount == 0 || surroundCount == 0 || edgePixels == 0 || diskCount == 0)
            continue;
        
        //diff betw surroundign cloth and ball brightness
        double contrast = surroundBrightness / surroundCount - innerBrightness/ innerCount;
        double edgeFraction = double(edgeCount) / edgePixels; // frac of px that are edges
        double foregroundFraction = double(foregroundCount) / diskCount;

        //only keep circles that look like dark balls
        //TUNE THE TH!!!!
        if (contrast > 50.0 && edgeFraction > 0.14 &&
            foregroundFraction > 0.20 && foregroundFraction < 0.50) {

            result.push_back(ball);
        }

    }//end of loop through weak circles
    
    return result;
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

//fraction of dark px in one ball
float getDarkFraction(const Mat& hsv, Point2f centre, Size axes) {

     //make a mask around the ball
     Mat mask = Mat::zeros(hsv.size(), CV_8UC1);

     ellipse(mask, centre, axes, 0, 0, 360,
             Scalar(255), FILLED);

     //total pixels inside ball mask
     int total = countNonZero(mask);

     //avoid dividing by zero
     if (total == 0)
         return 0.0f;

     //find dark pixels using hsv brightness
     Mat darkPixels;

     inRange(hsv,
             Scalar(0, 0, 0),
             Scalar(180, 255, 59),
             darkPixels);

     //only keep dark pixels inside ball
     Mat darkInside;
     bitwise_and(darkPixels, mask, darkInside);

     //calculate fraction of ball that is dark
     return static_cast<float>(countNonZero(darkInside)) / total;
 }




//train model
vector<Mat> trainBallRecognition(vector <float> & referenceDarkFractions){
    //load the training image
    Mat trainingImage = imread("Pool/AllBalls.jpeg");


    if (trainingImage.empty()) {
        cout << "Could not load training image" << endl;
        return {};
    }

    //load labelled positions from csv
    vector<Point2f> referencePositions = loadGroundTruth("AllBalls.jpeg");
    //safety
    if (referencePositions.size() != 16) {
        cout << "Expected 16 training positions; found "
            << referencePositions.size() << endl;
        return {};
    }
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

        //also save how much of each ball is dark
        float darkFrac = getDarkFraction(hsv, centre, BALL_AXES);
        referenceDarkFractions.push_back(darkFrac);
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
vector<int> recogniseBalls(const Mat & transformed, const vector<Point2f> &ballPositions, const vector<Mat> & referenceHistogram, const vector<float> &referenceDarkFractions ) {
    

    //-1 if not recognised
    vector<int> predictedBalls(ballPositions.size(), -1);

    //check reference data
    if (referenceHistogram.size() != 16 || referenceDarkFractions.size() != 16){
        cout << "Missing reference data" << endl;
        return predictedBalls;
    }

    //convert transformed image to HSV
    Mat hsv;
    cvtColor(transformed, hsv, COLOR_BGR2HSV);

    const Size BALL_AXES(16, 30);

    int N = static_cast<int>(ballPositions.size());
    const int LABELS = 16;

    //store scores for each detected ball against all 16 reference balls
    vector<vector<float>> score(
        N, vector<float>(LABELS, 0.0f)
    );


    //recognise each detected ball
    for (int i = 0; i < N; ++i) {

        //histogram of detected ball
        Mat sampleHist = extractBallHist(hsv, ballPositions[i], BALL_AXES);


        //fraction of dark px in detected bal
        float sampleDarkFraction = getDarkFraction(hsv, ballPositions[i], BALL_AXES);

        for (int j = 0; j < LABELS; ++j) {
            //bhattacharyya distance, smaller = better match
            double distance = compareHist(sampleHist, referenceHistogram[j], HISTCMP_BHATTACHARYYA);

            //combine colout similarity with diff in darkness - 0.9 weights
            score[i][j] = static_cast<float>(1.0 - distance) - 0.9f * abs(sampleDarkFraction - referenceDarkFractions[j]);
        }
    }
    

    //best possible match - greedy matching algo
    //store psosible match betwen detection and ball no
    struct BallMatch{
        float similarity;
        int candidate;
        int identity;
    };

    vector <BallMatch> matches;

    //get all possible matches
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < LABELS; ++j) {
            matches.push_back({score[i][j], i, j});
        }
        
        //sort matches highest scores fst
        sort(matches.begin(), matches.end(), [](const BallMatch & a, const BallMatch & b)){

            if (a.similarity != b.similarity)
                return a.candidate > b.candidate;
            
            if (a.candidate != b.candidate)
                return a.candidate > b.candidate;

            return a.identity > b.identity;
        }

        //-1 for no detection assinged
        vector<int> assignment(N, -1);

        //track ball nos used
        vector<bool> used(LABELS, false);

        //go throgh best matches first
        for (const BallMatch & match : matches){

            //ignore with neg score
            if(match.similarity <= 0)
                break;

            //skip if alr used
            if (assignment[match.candidate] != -1 || used[match.identity])
                continue;
            
            //assign this ball number to the candidate
            assignment[match.candidate] = match.identity;
            used[match.identity] = true;
        }

        for(int i = 0; i < N; ++i){
            int j = assignment[i];
            predictedBalls[i] = (j == 15) ? 0 : (j >= 0 ? j + 1 : -1);
        }
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

//load th gt with ball labels
vector<pair<int, Point2f>> loadGroundTruthLabels(const string& imageName) {
    vector<pair<int, Point2f>> labelled;

    //open ground truth csv
    ifstream file("Ball Locations In Transformed Image (Second correction).csv");


    if (!file.is_open())
        return labelled;

    string line;

    //skip first two rows
    getline(file, line);
    getline(file, line);

    //find correct image
    while (getline(file, line)) {

        stringstream ss(line);
        string filename;

        getline(ss, filename, ',');

        if (filename != imageName)
            continue;

        //read coordinates for each of the 16 balls
        for (int i = 0; i < 16; ++i) {

            string row, col;

            getline(ss, row, ',');
            getline(ss, col, ',');

            //only include balls with coordinates
            if (!row.empty() && !col.empty()) {

                //white ball is 0, others are 1-15
                int label = (i == 15) ? 0 : i + 1;

                //store ball number with its position
                labelled.push_back({
                    label,
                    Point2f(stof(col), stof(row))
                });
            }
        }

        break;
    }

    return labelled;

}
//predicted positions vs gt
vector<int> evalDetection(const vector<Point2f> & tablePositions, const vector<Point2f> & groundTruth, const vector<int> & groundTruthLabels) {
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

            trueLabels[i] = groundTruthLabels[j];

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

    //skip false pos with no true identity
    if (trueLabels[i] < 0)
        continue;

    //only add valid predictions to confusion matrix
    if (predicted[i] >= 0 && predicted[i] <= 15) {
        confusion.at<int>(trueLabels[i], predicted[i])++;
    }

    //count matched balls even if not recognised
    total++;

    //check if identity is correct
    if (predicted[i] == trueLabels[i])
        correct++;
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

    //train once using AllBalls.jpeg
    vector<float> referenceDarkFractions;
    vector<Mat> referenceHistograms =
        trainBallRecognition(referenceDarkFractions);

    //check all 16 training balls loaded
    if (referenceHistograms.size() != 16 ||
        referenceDarkFractions.size() != 16) {

        cout << "Skipping recognition: incomplete training data" << endl;
        return;
    }
    //test Tables 1 to 6
    for (int imageNumber = 1; imageNumber <= 6; ++imageNumber) {

        string imageName = "Table" + to_string(imageNumber) + ".jpeg";

        cout << "\n========== " << imageName << " ==========\n";

        //load current table image
        Mat original = imread("Pool/" + imageName);

        if (original.empty()) {
            cout << "Could not open or find " << imageName << endl;
            continue;
        }


	//detect possible circles
	vector<Vec3f> potential = findBalls(original);

	//segment table
	Mat fgMask = segmentTable(original);

	//filter potential balls
	vector<Vec3f> ballPositions = filterBalls(potential, fgMask);
    //second go to recover dark balls missed
    ballPositions = recoverDarkBalls(original, ballPositions, fgMask);

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
    warpPerspective(original, transformed, perspective_matrix, WARP_SIZE);

    //recognise using trained histograms and dark fractions
    vector<int> predictedBalls =  recogniseBalls(transformed, tablePositions, referenceHistograms, referenceDarkFractions);

    //make a copy for drawing so recognition uses the clean image
    Mat transformedResult = transformed.clone();

    //draw transformed ball centres
     //draw positions and predicted ball numbers
    for (size_t i = 0; i < tablePositions.size(); ++i) {

        const Point2f& centre = tablePositions[i];

        circle(transformedResult, centre, 4,
            Scalar(0, 0, 255), FILLED);

        //show white ball as W
        string label = predictedBalls[i] == 0
                    ? "W" : to_string(predictedBalls[i]);

        putText(transformedResult, label,
                centre + Point2f(5, -8),
                FONT_HERSHEY_SIMPLEX, 0.45,
                Scalar(255, 255, 255), 1);
    }

    //use separate result files for each table
    string prefix = "Table" + to_string(imageNumber) + "_";

    imwrite(prefix + "transformed_result.png", transformedResult);

    //draw detected circles on original image
    Mat filtered = original.clone();

    for (const Vec3f& ball : ballPositions) {

        circle(filtered,
            Point(cvRound(ball[0]), cvRound(ball[1])),
            cvRound(ball[2]),
            Scalar(0, 0, 255), 2);
    }

    //save results
    imwrite(prefix + "filtered_result.png", filtered);
    imwrite(prefix + "foreground_mask.png", fgMask);

    //print detections and transformed positions
    cout << "Hough candidates: " << potential.size() << endl;
    cout << "Filtered candidates: " << ballPositions.size() << endl;
    cout << "Candidate\tBall\tRow\tColumn\n";

    for (size_t i = 0; i < tablePositions.size(); ++i) {

    cout << i + 1 << "\t\t"
        << predictedBalls[i] << "\t"
        << cvRound(tablePositions[i].y) << "\t"
        << cvRound(tablePositions[i].x) << endl;
    }

    //load gt for this table
    vector<pair<int, Point2f>> labelledGT = loadGroundTruthLabels(imageName);

    if (!labelledGT.empty()) {

        vector<Point2f> groundTruth;
        vector<int> labels;

        //split labels and positions
        for (const auto& item : labelledGT) {
            labels.push_back(item.first);
            groundTruth.push_back(item.second);
        }

        //evaluate detected positions
        vector<int> trueLabels = evalDetection(tablePositions, groundTruth, labels);
       
        //evaluate predicted identities
        evalRecognition(predictedBalls, trueLabels);

    } else {

        cout << "Skipping evaluation: no ground truth loaded for "
            << imageName << endl;
    }
}
}

int main() {
	MyApplication();
	return 0;
}

//the transformation not being perfect, criticise gt
//for the mask nexct is to maybe reduce the area we are histogramming
//mask needs to be changed for saturation


//the balls in 3 and 4 too close to edge, need to test and tune
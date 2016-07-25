

#include "detectObject.h"       

// Search for objects such as faces in the image using the given parameters, storing the multiple cv::Rects into 'objects'.
// Can use Haar cascades or LBP cascades for Face Detection, or even eye, mouth, or car detection.
// Input is temporarily shrunk to 'scaledWidth' for much faster detection, since 200 is enough to find faces.
void detectObjectsCustom(const Mat &img, CascadeClassifier &cascade,
	                                     vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize,
	                                     float searchScaleFactor, int minNeighbors)
{
    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, CV_BGR2GRAY);
    }
    else if (img.channels() == 4)
	{
        cvtColor(img, gray, CV_BGRA2GRAY);
    }
    else
	{
        gray = img;
    }
    Mat inputImg;
    float scale = img.cols / (float)scaledWidth;
    if (img.cols > scaledWidth) {
        int scaledHeight = cvRound(img.rows / scale);
        resize(gray, inputImg, Size(scaledWidth, scaledHeight));
    }
    else
	{
        inputImg = gray;
    }

	//直方图均衡化
    Mat equalizedImg;
    equalizeHist(inputImg, equalizedImg);

    // 在小的灰度图像中检测目标
    cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

    // 放大缩小后的图像
    if (img.cols > scaledWidth) 
	{
        for (int i = 0; i < (int)objects.size(); i++ ) {
            objects[i].x = cvRound(objects[i].x * scale);
            objects[i].y = cvRound(objects[i].y * scale);
            objects[i].width = cvRound(objects[i].width * scale);
            objects[i].height = cvRound(objects[i].height * scale);
        }
    }

    // 保证目标在图像里面而不是在图像外面或者边缘上
    for (int i = 0; i < (int)objects.size(); i++ ) {
        if (objects[i].x < 0)
            objects[i].x = 0;
        if (objects[i].y < 0)
            objects[i].y = 0;
        if (objects[i].x + objects[i].width > img.cols)
            objects[i].x = img.cols - objects[i].width;
        if (objects[i].y + objects[i].height > img.rows)
            objects[i].y = img.rows - objects[i].height;
    }
}


// Search for just a single object in the image, such as the largest face, storing the result into 'largestObject'.
// Can use Haar cascades or LBP cascades for Face Detection, or even eye, mouth, or car detection.
// Input is temporarily shrunk to 'scaledWidth' for much faster detection, since 200 is enough to find faces.
// Note: detectLargestObject() should be faster than detectManyObjects().
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth)
{
    // Only search for just 1 object (the biggest in the image).
    int flags = CASCADE_FIND_BIGGEST_OBJECT;// | CASCADE_DO_ROUGH_SEARCH;
    // Smallest object size.
    Size minFeatureSize = Size(20, 20);
    // How detailed should the search be. Must be larger than 1.0.
    float searchScaleFactor = 1.1f;
    // How much the detections should be filtered out. This should depend on how bad false detections are to your system.
    // minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
    int minNeighbors = 4;

    // Perform Object or Face Detection, looking for just 1 object (the biggest in the image).
    vector<Rect> objects;
    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
    if (objects.size() > 0) {
        // 仅仅返回检测到的目标
        largestObject = (Rect)objects.at(0);
    }
    else {
        // 返回无效的矩形
        largestObject = Rect(-1,-1,-1,-1);
    }
}

// Search for many objects in the image, such as all the faces, storing the results into 'objects'.
// Can use Haar cascades or LBP cascades for Face Detection, or even eye, mouth, or car detection.
// Input is temporarily shrunk to 'scaledWidth' for much faster detection, since 200 is enough to find faces.
// Note: detectLargestObject() should be faster than detectManyObjects().
void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth)
{
    // Search for many objects in the one image.
    int flags = CASCADE_SCALE_IMAGE;

    // Smallest object size.
    Size minFeatureSize = Size(20, 20);
    // How detailed should the search be. Must be larger than 1.0.
    float searchScaleFactor = 1.1f;
    // How much the detections should be filtered out. This should depend on how bad false detections are to your system.
    // minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
    int minNeighbors = 4;

    // Perform Object or Face Detection, looking for many objects in the one image.
    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
}

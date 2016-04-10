#ifndef FEATURES_H_
#define FEATURES_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "complexmat.hpp"

class Features
{
public:
	Features(){}
	Features(bool useHOGFeature, bool useYUVFeature, bool useCNNFeature):useHOGFeature(useHOGFeature), useYUVFeature(useYUVFeature){}
	void setHOGPara(int HOGBinSize, int HOGOriented, int HOGType);
	void setCNNPara(int KernelSize, int KernelNum);
	void extractFeatures(cv::Mat & img);
	void setFeatures(bool useHOGFeature, bool useYUVFeature, bool useCNNFeature);
	std::vector<std::vector<int>> getFeatureSize();
	std::vector<cv::Mat> getFeature();

private:
	bool useHOGFeature = true;
	int HOGBinSize = 4; //Window size of HOG
	int HOGOriented = 9;
	int HOGType = 2; //0 = Normalized hist of gradients | 1 = HOG | 2 = FHOG | 3 = 0 && 1

	bool useYUVFeature = false;

	bool useCNNFeature = false;
	int KernelSize = 16;
	int KernelNum = 6;

	std::vector<std::vector<int>> featureSize;
	std::vector<cv::Mat> feature;

	std::vector<cv::Mat> extractHOGFeatures(cv::Mat & img);
	std::vector<cv::Mat> extractYUVFeatures(cv::Mat & img);
	std::vector<cv::Mat> extractCNNFeatures(cv::Mat & img);
	std::vector<cv::Mat> extractHOG(const cv::Mat & img, int use_hog, int bin_size, int n_orients, int soft_bin, float clip);
};

#endif
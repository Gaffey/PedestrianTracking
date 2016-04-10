#ifndef KCF_H_
#define KCF_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "complexmat.hpp"
#include "extractfeatures.h"

struct boundingBox
{
    double cx, cy, w, h;
};

class KCF
{
public:
	KCF(){}
	KCF(double gaussSigma, double outputSigmaFactor, double lambda, int *win):
		gaussSigma(gaussSigma), outputSigmaFactor(outputSigmaFactor), lambda(lambda)
		{
			windowSize[0] = win[0];
			windowSize[1] = win[1];
		}
	std::vector<cv::Mat> process(Features input, cv::Mat & img, int sizeX = 0, int sizeY = 0);
	void setTracker(Features input, cv::Mat & img, boundingBox & bbox, bool boxChanged);
	void setSearchWindow(int windowSizeX, int windowSizeY);
	void updateModel(double updateStep);
	void initTracker(Features input, cv::Mat & img, boundingBox & groundtruth);
	int getSizeX();
	int getSizeY();
	
private:
	double lambda = 1e-4;
	double gaussSigma = 0.5;
	std::vector<double> outputSigma;
	double outputSigmaFactor = 0.1;

	int windowSize[2];
	
	std::vector<cv::Mat> cosWindow;

	boundingBox kcfBox;

	std::vector<ComplexMat> yF;
    std::vector<ComplexMat> alphaF;
    std::vector<ComplexMat> xF;
    std::vector<ComplexMat> alphaFNew;
    std::vector<ComplexMat> xFNew;

    cv::Mat getSubwindow(const cv::Mat & input, int cx, int cy);
    cv::Mat gaussianShapedLabels(double sigma, int dim1, int dim2);
    ComplexMat gaussianCorrelation(const ComplexMat & xf, const ComplexMat & yf, double sigma, bool autoCorrelation);
    cv::Mat circShift(const cv::Mat & patch, int xRot, int yRot);
    cv::Mat getCosWindow(int dim1, int dim2);
    ComplexMat fft2(const cv::Mat & input);
    ComplexMat fft2(const cv::Mat & input, const cv::Mat & cosWindow);
    cv::Mat ifft2(const ComplexMat & inputF);
};

#endif
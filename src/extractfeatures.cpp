#include "extractfeatures.h"
#include "gradientMex.h"
#include <opencv2/core/core_c.h>
#include <iostream>

void Features::setHOGPara(int HOGBinSize, int HOGOriented, int HOGType)
{
	this -> HOGBinSize = HOGBinSize;
	this -> HOGOriented = HOGOriented;
	this -> HOGType = HOGType;
}

void Features::setCNNPara(int KernelSize, int KernelNum)
{
	this -> KernelSize = KernelSize;
	this -> KernelNum = KernelNum;
}

void Features::extractFeatures(cv::Mat & img)
{
	feature.clear();
    featureSize.clear();
	if(useHOGFeature)
	{
		std::vector<cv::Mat> hogF = extractHOGFeatures(img);
		int hogFNum = hogF.size();
		for(int i = 0; i < hogFNum; i++)
		{
			feature.push_back(hogF[i]);
            std::vector<int> imgsize;
            imgsize.push_back(img.cols/HOGBinSize);
            imgsize.push_back(img.rows/HOGBinSize);
			featureSize.push_back(imgsize);
		}
	}
	if(useYUVFeature)
	{
		std::vector<cv::Mat> yuvF = extractYUVFeatures(img);
		int yuvFNum = yuvF.size();
		for(int i = 0; i < yuvFNum; i++)
		{
			feature.push_back(yuvF[i]);
            std::vector<int> imgsize;
            imgsize.push_back(img.cols);
            imgsize.push_back(img.rows);
			featureSize.push_back(imgsize);
		}
	}
	if(useCNNFeature)
	{
		std::vector<cv::Mat> cnnF = extractCNNFeatures(img);
		int cnnFNum = cnnF.size();
		for(int i = 0; i < cnnFNum; i++)
		{
			feature.push_back(cnnF[i]);
            std::vector<int> imgsize;
            imgsize.push_back(img.cols/KernelSize);
            imgsize.push_back(img.rows/KernelSize);
			featureSize.push_back(imgsize);
		}
	}
}

std::vector<std::vector<int>> Features::getFeatureSize()
{
	return featureSize;
}
	
std::vector<cv::Mat> Features::getFeature()
{
	return feature;
}

std::vector<cv::Mat> Features::extractHOGFeatures(cv::Mat & img)
{
    cv::Mat input;
    if(img.channels() == 3)
    {
        cv::cvtColor(img, input, CV_BGR2GRAY);
        input.convertTo(input, CV_32FC1);
    }
    else
        img.convertTo(input, CV_32FC1);
	if(HOGType < 3)
		return extractHOG(input, HOGType, HOGBinSize, HOGOriented, -1, 0.2);
	else
	{
		std::vector<cv::Mat> hogF1 = extractHOG(input, 0, HOGBinSize, HOGOriented, -1, 0.2);
		std::vector<cv::Mat> hogF2 = extractHOG(input, 1, HOGBinSize, HOGOriented, -1, 0.2);
		int f2Num = hogF2.size();
		for(int i = 0; i < f2Num; i++)
			hogF1.push_back(hogF2[i]);
		return hogF1;
	}
}

std::vector<cv::Mat> Features::extractYUVFeatures(cv::Mat & img)
{
    cv::Mat input;
    cv::cvtColor(img, input, CV_BGR2YUV);
    cv::Size yuvSize = input.size();
   	std::vector<cv::Mat> yuvF;
   	cv::split(input, yuvF);
    return yuvF;
}

std::vector<cv::Mat> Features::extractCNNFeatures(cv::Mat & img)
{
	std::vector<cv::Mat> v;
	return v;
}

std::vector<cv::Mat> Features::extractHOG(const cv::Mat & img, int use_hog = 2, int bin_size = 4, int n_orients = 9, int soft_bin = -1, float clip = 0.2)
{
    // d image dimension -> gray image d = 1
    // h, w -> height, width of image
    // full -> ??
    // I -> input image, M, O -> mag, orientation OUTPUT
    int h = img.rows, w = img.cols, d = 1;
    bool full = true;
    if (h < 2 || w < 2) {
        std::cerr << "I must be at least 2x2." << std::endl;
        return std::vector<cv::Mat>();
    }

//        //image rows-by-rows
//        float * I = new float[h*w];
//        for (int y = 0; y < h; ++y) {
//            const float * row_ptr = img.ptr<float>(y);
//            for (int x = 0; x < w; ++x) {
//                I[y*w + x] = row_ptr[x];
//            }
//        }

    //image cols-by-cols
    float * I = new float[h*w];
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            I[x*h + y] = img.at<float>(y, x)/255.f;
        }
    }

    float *M = new float[h*w], *O = new float[h*w];
    gradMag(I, M, O, h, w, d, full);

    int n_chns = (use_hog == 0) ? n_orients : (use_hog==1 ? n_orients*4 : n_orients*3+5);
    int hb = h/bin_size, wb = w/bin_size;

    float *H = new float[hb*wb*n_chns];
    memset(H, 0, hb*wb*n_chns*sizeof(float));

    if (use_hog == 0) {
        full = false;   //by default
        gradHist( M, O, H, h, w, bin_size, n_orients, soft_bin, full );
    } else if (use_hog == 1) {
        full = false;   //by default
        hog( M, O, H, h, w, bin_size, n_orients, soft_bin, full, clip );
    } else {
        fhog( M, O, H, h, w, bin_size, n_orients, soft_bin, clip );
    }

    //convert, assuming row-by-row-by-channel storage
    std::vector<cv::Mat> res;
    int n_res_channels = (use_hog == 2) ? n_chns-1 : n_chns;    //last channel all zeros for fhog
    res.reserve(n_res_channels);
    for (int i = 0; i < n_res_channels; ++i) {
        //output rows-by-rows
//            cv::Mat desc(hb, wb, CV_32F, (H+hb*wb*i));

        //output cols-by-cols
        cv::Mat desc(hb, wb, CV_32F);
        for (int x = 0; x < wb; ++x) {
            for (int y = 0; y < hb; ++y) {
                desc.at<float>(y,x) = H[i*hb*wb + x*hb + y];
            }
        }

        res.push_back(desc.clone());
    }

    //clean
    delete [] I;
    delete [] M;
    delete [] O;
    delete [] H;

    return res;
}
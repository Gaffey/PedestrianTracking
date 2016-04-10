#include "kcf.h"
#include <iostream>

std::vector<cv::Mat> KCF::process(Features input, cv::Mat & img, int sizeX, int sizeY)
{
	cv::Mat newImg = getSubwindow(img, kcfBox.cx, kcfBox.cy);
	cv::Mat scaleImg = newImg;
	if(sizeX != 0) cv::resize(newImg, scaleImg, cv::Size(sizeX, sizeY), 0, 0, 0);
	input.extractFeatures(scaleImg);
	std::vector<cv::Mat> feature = input.getFeature();
	std::vector<std::vector<int>> featureSize = input.getFeatureSize();
	std::vector<cv::Mat> response;
	int featureNum = feature.size();
	for(int i = 0; i < featureNum; i++)
	{
	    ComplexMat zf = fft2(feature[i], cosWindow[i]);
	    ComplexMat kzf = gaussianCorrelation(zf, xF[i], gaussSigma, false);
	    response.push_back(ifft2(alphaF[i] * kzf));
	}
	return response;
}

void KCF::setTracker(Features input, cv::Mat & img, boundingBox & bbox, bool boxChanged = false)
{
    kcfBox = bbox;
    windowSize[0] = kcfBox.w * 2.5;
    windowSize[1] = kcfBox.h * 2.5;
    //what if bbox size changed? shall we change window size?
    //
    //work to do here
    //
    //
    cv::Mat newImg = getSubwindow(img, kcfBox.cx, kcfBox.cy);
    input.extractFeatures(newImg);
	std::vector<cv::Mat> feature = input.getFeature();
	std::vector<std::vector<int>> featureSize = input.getFeatureSize();
	std::vector<cv::Mat> response;
	int featureNum = feature.size();
	if(boxChanged)
		for(int i = 0; i < featureNum; i++)
		{
			outputSigma[i] = (std::sqrt(bbox.w*bbox.h) * outputSigmaFactor / (windowSize[0] / featureSize[i][0]));
		    //window weights, i.e. labels
		    yF[i] = (fft2(gaussianShapedLabels(outputSigma[i], featureSize[i][0], featureSize[i][1])));
		    cosWindow[i] = (getCosWindow(yF[i].cols, yF[i].rows));
		}
	for(int i = 0; i < featureNum; i++)
	{
	    //obtain a sub-window for training initial model
	    xFNew[i] = fft2(feature[i], cosWindow[i]);
	    //Kernel Ridge Regression, calculate alphas (in Fourier domain)
	    ComplexMat kf = gaussianCorrelation(xFNew[i], xFNew[i], gaussSigma, true);
	    alphaFNew[i] = yF[i] / (kf + lambda);
	}
}

int KCF::getSizeX()
{
    return windowSize[0];
}

int KCF::getSizeY()
{
    return windowSize[1];
}

void KCF::initTracker(Features input, cv::Mat & img, boundingBox & groundtruth)
{
    kcfBox = groundtruth;
    windowSize[0] = kcfBox.w * 2.5;
    windowSize[1] = kcfBox.h * 2.5;
    cv::Mat newImg = getSubwindow(img, groundtruth.cx, groundtruth.cy);
    input.extractFeatures(newImg);
	std::vector<cv::Mat> feature = input.getFeature();
	std::vector<std::vector<int>> featureSize = input.getFeatureSize();
	int featureNum = feature.size();
	for(int i = 0; i < featureNum; i++)
	{
		outputSigma.push_back(std::sqrt(groundtruth.w*groundtruth.h) * outputSigmaFactor / (windowSize[0] / featureSize[i][0]));
        //window weights, i.e. labels
	    yF.push_back(fft2(gaussianShapedLabels(outputSigma[i], featureSize[i][0], featureSize[i][1])));
        cosWindow.push_back(getCosWindow(yF[i].cols, yF[i].rows));
	    //obtain a sub-window for training initial model
        ComplexMat xFget = fft2(feature[i], cosWindow[i]);
	    xF.push_back(xFget);
	    xFNew.push_back(xFget);
	    //Kernel Ridge Regression, calculate alphas (in Fourier domain)
	    ComplexMat kf = gaussianCorrelation(xFNew[i], xFNew[i], gaussSigma, true);
	    ComplexMat alphaFget = yF[i] / (kf + lambda);
	    alphaF.push_back(alphaFget);
	    alphaFNew.push_back(alphaFget);
	}
}

void KCF::setSearchWindow(int windowSizeX, int windowSizeY)
{
	windowSize[0] = windowSizeX;
	windowSize[1] = windowSizeY;
}

void KCF::updateModel(double updateStep)
{
	int size = xF.size();
	for(int i = 0; i < size; i++)
	{
		xF[i] = xF[i] * (1.0 - updateStep) + xFNew[i] * updateStep;
		alphaF[i] = alphaF[i] * (1.0 - updateStep) + alphaFNew[i] * updateStep;
	}
}

cv::Mat KCF::gaussianShapedLabels(double sigma, int dim1, int dim2)
{
    cv::Mat labels(dim2, dim1, CV_32FC1);
    int range_y[2] = {-dim2 / 2, dim2 - dim2 / 2};
    int range_x[2] = {-dim1 / 2, dim1 - dim1 / 2};

    double sigma_s = sigma*sigma;

    for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j){
        float * row_ptr = labels.ptr<float>(j);
        double y_s = y*y;
        for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i){
            row_ptr[i] = std::exp(-0.5 * (y_s + x*x) / sigma_s);
        }
    }
    //rotate so that 1 is at top-left corner (see KCF paper for explanation)
    cv::Mat rot_labels = circShift(labels, range_x[0], range_y[0]);
    //sanity check, 1 at top left corner
    assert(rot_labels.at<float>(0,0) >= 1.f - 1e-10f);

    return rot_labels;
}

cv::Mat KCF::circShift(const cv::Mat &patch, int xRot, int yRot)
{
    cv::Mat rot_patch(patch.size(), CV_32FC1);
    cv::Mat tmp_xRot(patch.size(), CV_32FC1);

    //circular rotate x-axis
    if (xRot < 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(-xRot, patch.cols);
        cv::Range rot_range(0, patch.cols - (-xRot));
        patch(cv::Range::all(), orig_range).copyTo(tmp_xRot(cv::Range::all(), rot_range));

        //rotated part
        orig_range = cv::Range(0, -xRot);
        rot_range = cv::Range(patch.cols - (-xRot), patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_xRot(cv::Range::all(), rot_range));
    }else if (xRot > 0){
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols - xRot);
        cv::Range rot_range(xRot, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_xRot(cv::Range::all(), rot_range));

        //rotated part
        orig_range = cv::Range(patch.cols - xRot, patch.cols);
        rot_range = cv::Range(0, xRot);
        patch(cv::Range::all(), orig_range).copyTo(tmp_xRot(cv::Range::all(), rot_range));
    }
    //circular rotate y-axis
    if (yRot < 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(-yRot, patch.rows);
        cv::Range rot_range(0, patch.rows - (-yRot));
        tmp_xRot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        //rotated part
        orig_range = cv::Range(0, -yRot);
        rot_range = cv::Range(patch.rows - (-yRot), patch.rows);
        tmp_xRot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }else if (yRot > 0){
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows - yRot);
        cv::Range rot_range(yRot, patch.rows);
        tmp_xRot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        //rotated part
        orig_range = cv::Range(patch.rows - yRot, patch.rows);
        rot_range = cv::Range(0, yRot);
        tmp_xRot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }

    return rot_patch;
}

ComplexMat KCF::fft2(const cv::Mat &input)
{
    cv::Mat complex_result;
//    cv::Mat padded;                            //expand input image to optimal size
//    int m = cv::getOptimalDFTSize( input.rows );
//    int n = cv::getOptimalDFTSize( input.cols ); // on the border add zero pixels
//    copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
//    cv::dft(padded, complex_result, cv::DFT_COMPLEX_OUTPUT);
//    return ComplexMat(complex_result(cv::Range(0, input.rows), cv::Range(0, input.cols)));

    cv::dft(input, complex_result, cv::DFT_COMPLEX_OUTPUT);
    return ComplexMat(complex_result);
}

ComplexMat KCF::fft2(const cv::Mat &input, const cv::Mat &cosWindow)
{
    ComplexMat result(input.rows, input.cols, 1);
    cv::Mat complex_result;
    cv::dft(input.mul(cosWindow), complex_result, cv::DFT_COMPLEX_OUTPUT);
    result.set_channel(0, complex_result);

    return result;
}

cv::Mat KCF::ifft2(const ComplexMat &inputF)
{

    cv::Mat real_result;
    if (inputF.n_channels == 1){
        cv::dft(inputF.to_cv_mat(), real_result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    } else {
        std::vector<cv::Mat> mat_channels = inputF.to_cv_mat_vector();
        std::vector<cv::Mat> ifft_mats(inputF.n_channels);
        for (int i = 0; i < inputF.n_channels; ++i) {
            cv::dft(mat_channels[i], ifft_mats[i], cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        }
        cv::merge(ifft_mats, real_result);
    }
    return real_result;
}

//hann window actually (Power-of-cosine windows)
cv::Mat KCF::getCosWindow(int dim1, int dim2)
{
    cv::Mat m1(1, dim1, CV_32FC1), m2(dim2, 1, CV_32FC1);
    double N_inv = 1./(static_cast<double>(dim1)-1.);
    for (int i = 0; i < dim1; ++i)
        m1.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
    N_inv = 1./(static_cast<double>(dim2)-1.);
    for (int i = 0; i < dim2; ++i)
        m2.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
    cv::Mat ret = m2*m1;
    return ret;
}

// Returns sub-window of image input centered at [cx, cy] coordinates),
// with size [width, height]. If any pixels are outside of the image,
// they will replicate the values at the borders.
cv::Mat KCF::getSubwindow(const cv::Mat & input, int cx, int cy)
{
    cv::Mat patch;
    int width = windowSize[0];
    int height = windowSize[1];

    int x1 = cx - width/2;
    int y1 = cy - height/2;
    int x2 = cx + width/2;
    int y2 = cy + height/2;

    //out of image
    if (x1 >= input.cols || y1 >= input.rows || x2 < 0 || y2 < 0) {
        patch.create(height, width, CV_32FC1);
        patch.setTo(0.f);
        return patch;
    }

    int top = 0, bottom = 0, left = 0, right = 0;

    //fit to image coordinates, set border extensions;
    if (x1 < 0) {
        left = -x1;
        x1 = 0;
    }
    if (y1 < 0) {
        top = -y1;
        y1 = 0;
    }
    if (x2 >= input.cols) {
        right = x2 - input.cols + width % 2;
        x2 = input.cols;
    } else
        x2 += width % 2;

    if (y2 >= input.rows) {
        bottom = y2 - input.rows + height % 2;
        y2 = input.rows;
    } else
        y2 += height % 2;

    if (x2 - x1 == 0 || y2 - y1 == 0)
        patch = cv::Mat::zeros(height, width, CV_32FC1);
    else
        cv::copyMakeBorder(input(cv::Range(y1, y2), cv::Range(x1, x2)), patch, top, bottom, left, right, cv::BORDER_REPLICATE);

    //sanity check
    assert(patch.cols == width && patch.rows == height);

    return patch;
}

ComplexMat KCF::gaussianCorrelation(const ComplexMat & xf, const ComplexMat & yf, double sigma, bool autoCorrelation = false)
{
    float xf_sqr_norm = xf.sqr_norm();
    float yf_sqr_norm = autoCorrelation ? xf_sqr_norm : yf.sqr_norm();

    ComplexMat xyf = autoCorrelation ? xf.sqr_mag() : xf * yf.conj();

    //ifft2 and sum over 3rd dimension, we dont care about individual channels
    cv::Mat xy_sum(xf.rows, xf.cols, CV_32FC1);
    xy_sum.setTo(0);
    cv::Mat ifft2_res = ifft2(xyf);
    for (int y = 0; y < xf.rows; ++y) {
        float * row_ptr = ifft2_res.ptr<float>(y);
        float * row_ptr_sum = xy_sum.ptr<float>(y);
        for (int x = 0; x < xf.cols; ++x){
            row_ptr_sum[x] = std::accumulate((row_ptr + x*ifft2_res.channels()), (row_ptr + x*ifft2_res.channels() + ifft2_res.channels()), 0.f);
        }
    }

    float numel_xf_inv = 1.f/(xf.cols * xf.rows * xf.n_channels);
    cv::Mat tmp;
    cv::exp(- 1.f / (sigma * sigma) * cv::max((xf_sqr_norm + yf_sqr_norm - 2 * xy_sum) * numel_xf_inv, 0), tmp);

    return fft2(tmp);
}
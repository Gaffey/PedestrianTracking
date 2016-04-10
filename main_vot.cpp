#include <stdlib.h>

#include "kcf.h"
#include "extractfeatures.h"
#include "vot.hpp"
#include <iostream>

int main()
{
    //load region, images and prepare for output
    VOT vot_io("region.txt", "images.txt", "output.txt");

    KCF tracker;
    cv::Mat image;
    Features input;

    //img = firts frame, initPos = initial position in the first frame
    cv::Rect init_rect = vot_io.getInitRectangle();
    vot_io.outputBoundingBox(init_rect);
    vot_io.getNextImage(image);

    boundingBox bb;
    bb.cx = init_rect.x + init_rect.width/2.;
    bb.cy = init_rect.y + init_rect.height/2.;
    bb.w = init_rect.width;
    bb.h = init_rect.height;
    tracker.initTracker(input, image, bb);

    double avg_time = 0.;
    int frames = 0;
    while (vot_io.getNextImage(image) == 1){
        double time_profile_counter = cv::getCPUTickCount();
        std::vector<cv::Mat> trackResult = tracker.process(input, image);
        time_profile_counter = cv::getCPUTickCount() - time_profile_counter;
        //std::cout << "  -> speed : " <<  time_profile_counter/((double)cvGetTickFrequency()*1000) << "ms. per frame" << std::endl;
        avg_time += time_profile_counter/((double)cvGetTickFrequency()*1000);
        frames++;
        int trnum = trackResult.size();
        cv::Mat result = trackResult[0];
        for(int i = 1; i < trnum; i++)
        {
            result += trackResult[i];   
        }
        std::cout<<"result:"<<std::endl;
        std::cout<<result<<std::endl;
        double min_val, max_val;
        cv::Point2i min_loc, max_loc;
        cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
        if (max_loc.y > result.rows/2) //wrap around to negative half-space of vertical axis
            max_loc.y = max_loc.y - result.rows;
        if (max_loc.x > result.cols/2) //same for horizontal axis
            max_loc.x = max_loc.x - result.cols;
        bb.cx += 4 * max_loc.x;
        bb.cy += 4 * max_loc.y;

        std::cout<<"macloc:"<<max_loc.x<<" "<<max_loc.y<<std::endl;
        tracker.setTracker(input, image, bb, false);
        tracker.updateModel(0.02);
        vot_io.outputBoundingBox(cv::Rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h));

       // cv::rectangle(image, cv::Rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h), CV_RGB(0,255,0), 2);
       // cv::imshow("output", image);
       // cv::waitKey();

//        std::stringstream s;
//        std::string ss;
//        int countTmp = frames;
//        s << "imgs" << "/img" << (countTmp/10000);
//        countTmp = countTmp%10000;
//        s << (countTmp/1000);
//        countTmp = countTmp%1000;
//        s << (countTmp/100);
//        countTmp = countTmp%100;
//        s << (countTmp/10);
//        countTmp = countTmp%10;
//        s << (countTmp);
//        s << ".jpg";
//        s >> ss;
//        //set image output parameters
//        std::vector<int> compression_params;
//        compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
//        compression_params.push_back(90);
//        cv::imwrite(ss.c_str(), image, compression_params);
    }

    std::cout << "Average processing speed " << avg_time/frames <<  "ms. (" << 1./(avg_time/frames)*1000 << " fps)" << std::endl;

    return EXIT_SUCCESS;
}
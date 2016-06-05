#include <iostream>
#include <math.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Segmentation.h"
#include "Filter.h"

////////////////////////////////////////////////////////////////////////////////////
// constructor and destructor
////////////////////////////////////////////////////////////////////////////////////
Segmentation::Segmentation(){}

Segmentation::~Segmentation(){}

////////////////////////////////////////////////////////////////////////////////////
// Cut template image from input and save as file
////////////////////////////////////////////////////////////////////////////////////
void Segmentation::cutAndSave(const cv::Mat &input, cv::Point origin, cv::Size size, const cv::string &filename)
{
    // define rectangle from origin and size
    cv::Rect rect(origin, size);

    // cut the rectangle and create template
    cv::Mat templ = input(rect);

    // save template to file
    cv::imwrite(filename, templ);
}

////////////////////////////////////////////////////////////////////////////////////
// Compute normalized cross correlation function
////////////////////////////////////////////////////////////////////////////////////
void Segmentation::crossCorrelate(const cv::Mat &input, const cv::Mat &templ, cv::Mat &output)
{
    int rows = input.rows;
    int cols = input.cols;

    int tRows = templ.rows;
    int tCols = templ.cols;

    output.release();

    // create a float image for the output;
    output = cv::Mat((rows-tRows+1),(cols-tCols+1), CV_32F); // lets take the image output also to be oif the same 
    
    for (int r = 0 ; r < (rows-tRows) ; ++r)
    {
        for (int c = 0 ; c < (cols - tCols) ; ++c)
        {
            float product_template_image = 0 , sqrt_squared_template = 0 , sqrt_squared_image = 0 ,total_squared_image = 0 ,total_squared_template = 0,product_norm = 0 ;

            for (int i = 0 ; i < tRows ; ++i)
            {
                for (int j = 0 ; j < tCols ; ++j)
                {
                    //statistics calculations                     
                    //Pixel wise processing
                    product_template_image += (input.at<float>(r+i,c+j) * templ.at<float>(i,j));
                    //template normalisation factor         
                    total_squared_template += pow(templ.at<float>(i,j),2);
                    //image normalisation factor
                    total_squared_image += pow(input.at<float>(r+i,c+j),2);  
                }
            }
        
            //final formula calculation ... 
            output.at<float>(r,c) = product_template_image / (sqrt(total_squared_template) * sqrt(total_squared_image));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////
// Find brightest pixel and return its coordinates as Point
////////////////////////////////////////////////////////////////////////////////////
cv::Point Segmentation::findMaximum(const cv::Mat &input)
{
    // declare array to hold the indizes
    int maxIndex[2];

    // find the maximum
    cv::minMaxIdx(input, 0, 0, 0, maxIndex);

    // create Point and return
    return cv::Point(maxIndex[1], maxIndex[0]);
}

////////////////////////////////////////////////////////////////////////////////////
// Add a black rectangle to an image
////////////////////////////////////////////////////////////////////////////////////
void Segmentation::drawRect(const cv::Mat &input, cv::Point origin, cv::Size size, cv::Mat &output)
{
    // define rectangle from origin and size
    cv::Rect rect(origin, size);

    // copy input image to output
    output = input.clone();

    // draw the rectangle
    cv::rectangle(output, rect, 0, 2);
}


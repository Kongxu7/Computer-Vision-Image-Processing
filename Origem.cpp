#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <functional>
#include <vector>
#include <opencv2\opencv.hpp>
#include <conio.h>
#include <opencv\cvaux.h>
#include <algorithm>
#include <numeric>
#include <cstddef>

using namespace std;
using namespace cv;

int val_Red = 0;
int val_Green = 0;
int val_Blue = 0;
Mat result;
Mat Iii;

void on_trackbar(int, void*)
{
    float SH = 0.1; 

    float cr_val = (float)val_Red / 255.0;
    float mg_val = (float)val_Green / 255.0;
    float yb_val = (float)val_Blue / 255.0;
   
    float R1 = 1;
    float G1 = 0;
    float B1 = 0;

    float R2 = 2;
    float G2 = 0;
    float B2 = 0;

    /*float DR = (R1 + R2) - 0.5;
      float DG = (G1+G2) -0.5;
      floar DB = (B1+B2) - 0.5; */
    float DR = (1 - cr_val) * R1 + (cr_val)*R2 - 0.5;
    float DG = (1 - cr_val) * G1 + (cr_val)*G2 - 0.5;
    float DB = (1 - cr_val) * B1 + (cr_val)*B2 - 0.5;

        result = Iii + (Scalar(DB, DG, DR) * SH);

   
    R1 = 0;
    G1 = 1;
    B1 = 0;

    R2 = 0;
    G2 = 2;
    B2 = 0;

        /*float DR = (R1 + R2) - 0.5;
        float DG = (G1+G2) -0.5;
        floar DB = (B1+B2) - 0.5; */
    DR = (1 - mg_val) * R1 + (mg_val)*R2 - 0.5;
    DG = (1 - mg_val) * G1 + (mg_val)*G2 - 0.5;
    DB = (1 - mg_val) * B1 + (mg_val)*B2 - 0.5;

       result += (Scalar(DB, DG, DR) * SH);

  

    R1 = 0;
    G1 = 0;
    B1 = 1;

    R2 = 0;
    G2 = 0;
    B2 = 2;

        /*float DR = (R1 + R2) - 0.5;
        float DG = (G1+G2) -0.5;
        floar DB = (B1+B2) - 0.5; */

    DR = (1 - yb_val) * R1 + (yb_val)*R2 - 0.5;
    DG = (1 - yb_val) * G1 + (yb_val)*G2 - 0.5;
    DB = (1 - yb_val) * B1 + (yb_val)*B2 - 0.5;

    result += (Scalar(DB, DG, DR) * SH);

    imshow("RGB intensifier", result);
   // waitKey(10);
}

// ---------------------------------
// 
// ---------------------------------



// ---------------------------------
// 
// ---------------------------------


int main(int argc, char** argv)
{
    namedWindow("Image", cv::WINDOW_NORMAL);
    namedWindow("RGB intensifier");

    Iii = imread("star.ppm", 1);
    Iii.convertTo(Iii, CV_32FC1, 1.0 / 255.0);

    createTrackbar("Red", "Image", &val_Red, 255, on_trackbar);
    createTrackbar("Green", "Image", &val_Green, 255, on_trackbar);
    createTrackbar("Blue", "Image", &val_Blue, 255, on_trackbar);

    imshow("Image", Iii);
    // waitKey(0);

    Mat kuva;

    kuva = cv::imread("ghgbmnhmhg.png");

    for (int k = 0; k < 50; k++) {

        for (int r = 0; r < kuva.rows; r++) {

            for (int c = 0; c < kuva.cols; c++) {

                kuva.at<Vec3b>(r, c)[0] = kuva.at<Vec3b>(r, c)[0] * 0.f;
                kuva.at<Vec3b>(r, c)[1] = kuva.at<Vec3b>(r, c)[1] * 1.03f;
                kuva.at<Vec3b>(r, c)[2] = kuva.at<Vec3b>(r, c)[2] / 1.02f;
            }
        }
        namedWindow("Kuva ", 9);
        imshow("Kuva ", kuva);
        waitKey(40);
    }

    Mat image = imread("star.ppm", 1);
    namedWindow("Original", 1);
    imshow("Original", image);

    Mat img = imread("star.pgm", 2);
    namedWindow("Original 2", 2);
    imshow("Original 2", img);



    Mat imgContraste;
    img.convertTo(imgContraste, -1, 2, 0);

    String comcontraste = "Contraste aumentado em 20%";
    namedWindow(comcontraste);
    imshow(comcontraste, imgContraste);
    imshow(comcontraste, imgContraste);

    /*
        vector<Mat> rgb;
        rgb.push_back(canaisRgb[1]);
        rgb.push_back(cc);

    } */

    //Mat image;

    vector<Mat> rgbChannels(3);
    split(image, rgbChannels);

    Mat vav, nvimg;
    vav = Mat::zeros(Size(image.cols, image.rows), CV_8UC1);


    {
        vector<Mat> channels;
        channels.push_back(vav); //BGR | 0, 1, 2
        channels.push_back(vav);
        channels.push_back(rgbChannels[2]);


        merge(channels, nvimg);
        namedWindow("Red", 1);
        imshow("Red", nvimg);
    }


    {
        vector<Mat> channels;
        channels.push_back(vav);
        channels.push_back(rgbChannels[1]);
        channels.push_back(vav);

        merge(channels, nvimg);
        namedWindow("Green", 1);
        imshow("Green", nvimg);
    }


    {
        vector<Mat> channels;
        channels.push_back(rgbChannels[0]);
        channels.push_back(vav);
        channels.push_back(vav);

        merge(channels, nvimg);
        namedWindow("Blue", 1);
        imshow("Blue", nvimg);
    }


    /*
      uchar blue = intensity.val[0];
      uchar green = intensity.val[1];
      */

    {

   /* int r_slider = 0;
    int g_slider = 0;
    int b_slider = 0;
    int r_max = 255, g_max = 255, b_max = 255;
    /* int rmin = 0, rmax = 0, bmin = 0, bmax = 0,
         gmin = 0, gmax = 0; */

    //namedWindow("trackbar panel");

    /* createTrackbar("r", "trackbar panel", &r_slider, 255);
    createTrackbar("g", "trackbar panel", &g_slider, 255);
    createTrackbar("b", "trackbar panel", &b_slider, 255);

    Mat imagem, output;

    //switch = "0: OFF 1: ON"
   /* int track;
    switch(track) {

    case 1:
        0;
        break;
    case 2:
        1;
        break;

    }
    createTrackbar(switch, imagem, 0, 1); */
   // imagem = imread("star.ppm", 11);
    //imshow("input", imagem);
   // output = imread("star.ppm", 12);

       // while(true) {

        // imshow("input", imagem);
        // int b;
        // b = getTrackbarPos('b', b_slider);
          /* int r, g, b;
         r = getTrackbarPos("r", "trackbar panel");
         g = getTrackbarPos("g", "trackbar panel");
         b = getTrackbarPos("b", "trackbar panel");
         */
         //setTrackbarPos("r", imagem);
      //  inRange(imagem, Scalar(r_slider, g_slider, b_slider), Scalar(r_max, g_max, b_max), output);

       // imshow("Result window", output);
       //  output.release();
       //  waitKey(0);
        // if (key == 27) { break; }


       // }
      //imagem.release(); 


            
   /*
       while (true)
       {
           r_slider = getTrackbarPos("r", "trackbar panel");
           g_slider = getTrackbarPos("g", "trackbar panel");
           b_slider = getTrackbarPos("b", "trackbar panel");




           for (int i = 0; i < imagem.rows; i++)
           {
               for (int j = 0; j < imagem.cols; j++)
               {

                   Vec3b color = imagem.at(Point(i, j));
                   imagem.at(i, j)[2] = imagem.at(i, j)[2] + r_slider;




               }
           }





           for (int i = 0; i < imagem.rows; i++)
           {
               for (int j = 0; j < imagem.cols; j++)
               {

                   Vec3b color = imagem.at(Point(i, j));
                   cv::imagem.at(i, j)[1] = imagem.at(i, j)[1] + g_slider;


               }
           }







           for (int i = 0; i < imagem.rows; i++)
           {
               for (int j = 0; j < imagem.cols; j++)
               {

                   Vec3b color = imagem.at(Point(i, j));
                   imagem.at(i, j)[0] = imagem.at(i, j)[0] + b_slider;


               }
           }



           imshow("Imagem", imagem);


           */



    }



    cv::Mat pic = imread("star.ppm");

    if (pic.empty()) {

        cout << "Nao foi possivel abrir a imagem. " << endl;
        cin.get();
        return 0;
    }


    Mat pic_77;
    GaussianBlur(pic, pic_77, Size(7, 7), 0);


    Mat pic_55;
    GaussianBlur(pic, pic_55, Size(5, 5), 0);


    String window_77 = " Blurred w/ 7 x 7 Gaussian Kernel";
    String window_55 = "Blurred w/ 5 x 5 Gaussian Kernel";


    namedWindow(window_77);
    namedWindow(window_55);


    imshow(window_77, pic_77);
    imshow(window_55, pic_55);

     

    // ---------------------------------
    // 
    // ---------------------------------




    cv::Mat ppimg;


    ppimg = cv::imread("star.ppm", IMREAD_GRAYSCALE);


    for (int m = 0; m < 50; m++) {

        for (int p = 0; p < ppimg.rows; p++) {

            for (int l = 0; l < ppimg.cols; l++) {

                ppimg.at<uchar>(p, l) = (uchar)ppimg.at<uchar>(p, l) * 0.98;
            }

        }

        cv::imshow("Darker", ppimg);
        cv::waitKey(40);
    }


    label:
    Mat frame;
    namedWindow("Video Player");

    VideoCapture cap("vid.mp4");
    if (!cap.isOpened()) {
        cout << "Nao foi possivel abrir o arquivo. " << endl;
        system("pause");
        cin.get();
        return 0;
    }

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            goto label;
        }
        imshow("Video Player", frame);
        char c = (char)waitKey(30);
        if (c == 27) {
            break;
        }
    }
    cap.release();

    /*  VideoCapture cap("vid.mp4");

      while (1)   {

          Mat frame;
           cap >> frame;

           if (frame.empty())
               break;

           namedWindow("Video Player");
           imshow("Video Player", frame);

           char c = (char)waitKey(30);


      }

      cap.release();


      */




    destroyAllWindows();



    return 0;

}


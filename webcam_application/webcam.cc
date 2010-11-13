// ==========================================================================
// Version 1.0
// ==========================================================================
// (C)opyright: 2010
//
//   Ulm University
// 
// Creator: Ferdinand Deger, <Firstname>.<Lastname>@gmail.com
// Creator: Helmut Sedding, <Firstname>@<Lastname>.net
// ==========================================================================
//  A Simple Camera Capture Framework to demonstrate the real time capacity
//  of the CUDA object recognition Framework. 
//  Open CV is used to access the webcam and to downscale its images
//  libsvm(2.9) by Copyright (c) 2000-2009 Chih-Chung Chang and Chih-Jen Lin
//    as the support vector machine
// ==========================================================================


#ifdef WEBCAM_GUARD

#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <stdlib.h>

#include "svm.h"
#include "object_recognition.h"
#ifndef Malloc
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#endif



void convertImage(float**, IplImage*);
void printNodes(svm_node*,int);
void showImage (IplImage*);

IplImage* webcam_temp ;
IplImage* webcam_temp1 ;
IplImage* webcam_gray ;

int predicted_timer;
CvFont font1;
char LiveMessage[30];

int MAX_SVM_LENGTH = 10000;

int main() {

    //Variables needed for MPP
    /** all patches saved in this variable, patches are saved sparse to ensure sufficient space */
    t_patches patches;
    /** numbers of categories, that should be distinguished in the multiclass */
    int numberOfCategories;
    /** number of orientations is classified by the chosen gaborefile */
    int norientations = 4;
    /** location of the gaborfile in the dat-format specified in util/io_matlab */
    char* gaborfile = "Gabor4.dat";
    /** location of the patchesfile */
    //  char* patchesfile = "test_patches";
    //  char* patchesfile = "stdtest_15pics_4075patches_rev5_rnd7892341";
    char* patchesfile = "480pnum_test_15pics_480patches_rev6_rnd86453061";
    //  char* patchesfile = "960pnum_test_15pics_960patches_rev6_rnd96011005";

    webcam_readPatches(&patches, patchesfile);

    const int number_of_patches = patches.npatchesTotal();

    webcam_S2_loadPatches(&patches);

    float resize_factor = 3.428;

    //float* floatImg = new float[640*480];
    float* floatImg = new float[int(640*480/(resize_factor*resize_factor))];
    int image_counter = 0;
    int messageDelay = 160;
    //Variables needed for OpenCV
    //IplImage* temp = cvCreateImage( cvSize(640, 480), IPL_DEPTH_8U, 1 );
    webcam_temp = cvCreateImage( cvSize(640, 480), IPL_DEPTH_8U, 1 );
    webcam_temp1= cvCreateImage( cvSize(int(640/resize_factor), int(480/resize_factor)), IPL_DEPTH_8U, 1 );
    webcam_gray = cvCreateImage( cvSize(int(640/resize_factor), int(480/resize_factor)), IPL_DEPTH_8U, 1 );

    CvCapture* capture = cvCaptureFromCAM(2);// CV_CAP_ANY );
    if( !capture ) {
        fprintf( stderr, "ERROR: capture is NULL \n" );
        getchar();
        return -1;
    }
    // Create a window in which the captured images will be presented
    cvNamedWindow( "mywindow", CV_WINDOW_AUTOSIZE );


    //Variables needed for SVM
    bool exitModel = false;
    svm_problem    svmProblem;
    svm_parameter  svmParameter;

    svm_node*      x_space;
    svm_model*     svmModel;

    svmParameter.svm_type = C_SVC;
    //  svmParameter.kernel_type = RBF;
    svmParameter.degree = 3;
    svmParameter.gamma = 0;	// 1/num_features
    svmParameter.coef0 = 0;
    svmParameter.nu = 0.5;
    svmParameter.cache_size = 100;
    //  svmParameter.C = 1;
    svmParameter.eps = 1e-3;
    svmParameter.p = 0.1;
    svmParameter.shrinking = 1;
    svmParameter.probability = 0;
    svmParameter.nr_weight = 0;
    svmParameter.weight_label = NULL;
    svmParameter.weight = NULL;
    svmParameter.kernel_type = 0;
    svmParameter.C           = 13;

    svmProblem.l  = 0;
    svmProblem.y = Malloc(double,MAX_SVM_LENGTH);
    svmProblem.x = Malloc(struct svm_node *,MAX_SVM_LENGTH);
    x_space =      Malloc(struct svm_node,number_of_patches+1);

    int predicted_category = -1;
    predicted_timer    = 70;
    sprintf(LiveMessage, "Init") ;
    cvInitFont( &font1, CV_FONT_VECTOR0, 1.3f, 1.3f, 0.0f,4 );

    // Show the image captured from the camera in the window and repeat
    while( 1 ) {
        // Get one frame
        IplImage* frame = cvQueryFrame( capture );
        if( !frame ) {
            fprintf( stderr, "ERROR: frame is null...\n" );
            getchar();
            break;
        }


        // Do not release the frame!

        //If ESC key pressed, Key=0x10001B under OpenCV 0.9.7(linux version),
        //remove higher bits using AND operator
        char keyboard = cvWaitKey(10) & 255;
        if( keyboard == 27 ) break;
        else if( keyboard == '1' || keyboard == '2' || keyboard == '3' || keyboard == '4' || keyboard == '5') {
            int category =  keyboard - '0';
            int NoPic     = svmProblem.l++;
            printf("appending Image %d to category %d\n",NoPic,category);
            convertImage(&floatImg, frame);
            sprintf(LiveMessage, "Appending Img") ;
            showImage (webcam_temp);
            svm_node*  temp_nodes;
            extractVector(&temp_nodes, &patches, gaborfile, floatImg, (unsigned int)webcam_gray->widthStep, (unsigned int)webcam_gray->height);
            //printNodes(temp_nodes,number_of_patches);
            svmProblem.y[NoPic] = category;
            svmProblem.x[NoPic] = &temp_nodes[0];
            sprintf(LiveMessage, "Appended Im %d, cat %d ",NoPic,category ) ;
            predicted_timer = messageDelay;
        }

        else if (keyboard == 's') {
            if(svmProblem.l < 5) {
                printf("NO TRAINING, because there are only %d pics",svmProblem.l);
                sprintf(LiveMessage, "To few Img to classify") ;
                predicted_timer = messageDelay;
                showImage (frame);
                continue;
            }
            printf("TRAIN SVM!! No of Pics %d",svmProblem.l);
            sprintf(LiveMessage, "Run SVM") ;
            showImage (webcam_temp);
            if(exitModel) {
                svm_destroy_model(svmModel);
            }
            const char *error_msg;
            error_msg = svm_check_parameter(&svmProblem, &svmParameter);
            if(error_msg)
            {
                fprintf(stderr,"Error: %s\n",error_msg);
                exit(1);
            }
            svmModel = svm_train(&svmProblem, &svmParameter);
            exitModel = true;
            sprintf(LiveMessage, "Run SVM...done") ;
            predicted_timer = messageDelay;

        }

        else if (keyboard == 'r') {
            if(!exitModel) {
                printf("You have to train first, before you can recognize objects\n");
                sprintf(LiveMessage, "Nothing Learned Yet" ) ;
                predicted_timer = messageDelay;
                showImage (frame);
                continue;

            }
            printf("Recognizing\n");
            convertImage(&floatImg, frame);
            sprintf(LiveMessage, "Predicting" ) ;
            showImage (webcam_temp);
            svm_node*  temp_nodes;
            extractVector(&temp_nodes, &patches, gaborfile, floatImg, (unsigned int)webcam_gray->widthStep, (unsigned int)webcam_gray->height);
            //printNodes(temp_nodes,number_of_patches);
            double prediction = svm_predict(svmModel, temp_nodes);
            printf("Prediction %f\n",prediction);
            predicted_category = prediction;
            sprintf(LiveMessage, "Predict cat %d ",predicted_category ) ;
            predicted_timer = messageDelay;
            free (temp_nodes);
        }

        //    int svm_save_model(const char *model_file_name, const struct svm_model *model);
        //    struct svm_model *svm_load_model(const char *model_file_name);

        else if( keyboard == 'm' ) {
            char* filename = new char[30];
            sprintf(filename, "category/image_%04d.tga", image_counter++);
            convertImage(&floatImg, frame);
            webcamToTGA(floatImg, filename, (unsigned int)webcam_gray->widthStep, (unsigned int)webcam_gray->height);
            delete filename;
        }
        else if( keyboard == 'l' ) {
            convertImage(&floatImg, frame);
            webcam_evalTime( &patches, gaborfile, floatImg, (unsigned int)webcam_gray->widthStep, (unsigned int)webcam_gray->height) ;
        }
        //else {}
        showImage (frame);
        //Paint the Image
    }

    //CLean-up
    delete(floatImg);

    //Clean up mpp
    webcam_S2_clearPatches();

    //Clean up SVM
    if(exitModel) {
        svm_destroy_model(svmModel);
    }
    svm_destroy_param(&svmParameter);

    //clean up OpenCV
    cvReleaseImage( &webcam_temp);
    cvReleaseImage( &webcam_gray );
    cvReleaseCapture( &capture );
    cvDestroyWindow( "mywindow" );
    return 0;
}


// Print Image, with the message - message shows up predticted_timer long
//
void showImage (IplImage* frame) {
    if  (predicted_timer > 0) {
        predicted_timer--; 
        cvPutText( frame, LiveMessage, cvPoint( 50, 50 ), &font1, CV_RGB(0,0,0) );//CV_RGB(55,15,255) );
    }
    cvShowImage( "mywindow", frame );
}


void printNodes(svm_node* my_nodes,int len) {
    printf("printNodes sizeof(svm_node)%lu, sizeof(svm_node*)%lu\n",sizeof(svm_node),sizeof(svm_node*));
    for (int i = 0; i<len; ++i) {
        printf("%d:%f\n",my_nodes[i].index,my_nodes[i].value);
    }
}

// Simple Image convertions, like rescale and graysacale
//
void convertImage(float** out, IplImage* in) {
    float* img = *out;
    cvCvtColor( in, webcam_temp, CV_BGR2GRAY );

    cvResize( webcam_temp, webcam_temp1);

    if(!cvSaveImage("testoutput.tif",webcam_temp1)) {
        printf("Could not save: %s\n","testoutput.tif");
    }
    cvConvertImage(webcam_temp1, webcam_gray, CV_CVTIMG_FLIP);
    printf("tif saved\n"); //debug prompt
    printf("w:%d, h:%d, widthStep:%d, imageSize:%d\n",webcam_gray->width,webcam_gray->height,webcam_gray->widthStep,webcam_gray->imageSize);

    //convertImage(data,gray);
    for(int i=0; i<webcam_gray->imageSize; ++i) {
        img[i]=((float)(webcam_gray->imageData[i])) / 255;
    }
}
#else
#include<iostream>
main(){
printf("No OpenCV for webcam available");
}
#endif



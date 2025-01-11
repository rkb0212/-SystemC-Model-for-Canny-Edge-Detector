/* source: http://marathon.csee.usf.edu/edge/edge_detection.html */
/* URL: ftp://figment.csee.usf.edu/pub/Edge_Comparison/source_code/canny.src */

/* ECPS 203 Assignment 5 solution */

/* off-by-one bugs fixed by Rainer Doemer, 10/05/23 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "systemc.h"
#include <sstream>

#define VERBOSE 0

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0
#define BOOSTBLURFACTOR 90.0
#ifndef M_PI
#define M_PI 3.14159265356789202346
#endif
#define SIGMA 0.6
#define TLOW 0.3
#define THIGH 0.8

#define COLS 1920
#define ROWS 1080
#define IMAGE_SIZE (COLS * ROWS)
#define VIDEONAME "Engineering"
#define IMG_IN "video/" VIDEONAME "%03d.pgm"
#define IMG_OUT VIDEONAME "%03d_edges.pgm"
#define IMG_NUM 30   /* number of images processed (1 or more) */
#define AVAIL_IMG 30 /* number of different image frames (1 or more) */
#define SET_STACK_SIZE set_stack_size(128 * 1024 * 1024);

#define WINSIZE 21

// Template struct to handle arrays of different data types and sizes
template <typename T, int SIZE>
struct Image_s
{
   T img[SIZE];
   Image_s()
   {
      for (int i = 0; i < SIZE; i++)
         img[i] = 0;
   }
   Image_s &operator=(const Image_s &copy)
   {
      for (int i = 0; i < SIZE; i++)
         img[i] = copy.img[i];
      return *this;
   }
   operator T *() { return img; }
   T &operator[](const int index) { return img[index]; }
};

typedef Image_s<unsigned char, IMAGE_SIZE> IMAGE;
typedef Image_s<float, WINSIZE> KIMAGE;  // Kernel image
typedef Image_s<float, IMAGE_SIZE> FIMAGE;  // Floating-point image
typedef Image_s<short, IMAGE_SIZE> SIMAGE;  // Short integer image
SC_MODULE(Stimulus) {
    sc_fifo_out<IMAGE> ImgOut;      // Output FIFO for image data
    sc_fifo_out<sc_time> TimeOut;   // Output FIFO for simulation time
    IMAGE Image;
    sc_time Frame_generation_Time;
    char infilename[70];
    std::string msg = "";
    void get_images(void) {
        for (int i = 0; i < IMG_NUM; i++) {
            int n = i % AVAIL_IMG;
            sprintf(infilename, IMG_IN, n + 1);

            if (VERBOSE)
                printf("Reading the image %s.\n", infilename);
            if (read_pgm_image(infilename, Image, ROWS, COLS) == 0) {
                fprintf(stderr, "Error reading the input image, %s.\n", infilename);
                exit(1);
            }

            // Send image to the platform
            ImgOut.write(Image);

            // Record and send current simulation time
            Frame_generation_Time = sc_time_stamp();
            TimeOut.write(Frame_generation_Time);
            msg = " | at Time: " + Frame_generation_Time.to_string() + " : | Sent frame no.: " + std::to_string(i);
            SC_REPORT_INFO(name(), msg.c_str());
        }
    }

    int read_pgm_image(const char *infilename, unsigned char *image, int rows, int cols) {
        FILE *fp;
        char buf[71];
        int r, c;

        if ((fp = fopen(infilename, "r")) == NULL) {
            fprintf(stderr, "Error reading the file %s in read_pgm_image().\n", infilename);
            return 0;
        }

        fgets(buf, 70, fp);
        if (strncmp(buf, "P5", 2) != 0) {
            fprintf(stderr, "The file %s is not in PGM format.\n", infilename);
            fclose(fp);
            return 0;
        }
        do {
            fgets(buf, 70, fp);
        } while (buf[0] == '#');
        sscanf(buf, "%d %d", &c, &r);
        do {
            fgets(buf, 70, fp);
        } while (buf[0] == '#');

        if ((unsigned)rows != fread(image, cols, rows, fp)) {
            fprintf(stderr, "Error reading the image data.\n");
            fclose(fp);
            return 0;
        }

        fclose(fp);
        return 1;
    }

    SC_CTOR(Stimulus) {
        SC_THREAD(get_images);
        SET_STACK_SIZE
    }
};
SC_MODULE(Monitor) {
    sc_fifo_in<IMAGE> ImgIn;       // Input FIFO for processed image
    sc_fifo_in<sc_time> TimeIn;    // Input FIFO for recorded time
    IMAGE EdgeImage;
    sc_time sent_time, received_Time, previous_received_Time, latency;
    std::string msg = "";

    char outfilename[128];

    void save_images(void) {
        for (int i = 0; i < IMG_NUM; i++) {
            // Read processed image and sent time
            ImgIn.read(EdgeImage);
            TimeIn.read(sent_time);

            // Calculate and print delay
            sc_time current_time = sc_time_stamp();
            double delay_ms = (current_time - sent_time).to_seconds() * 1000;
            printf("[%.3f ms] Monitor: Frame %d received with delay %.3f ms\n",
                   current_time.to_seconds() * 1000, i + 1, delay_ms);

            // Save the processed image
            sprintf(outfilename, IMG_OUT, (i % AVAIL_IMG) + 1);
            if (write_pgm_image(outfilename, EdgeImage, ROWS, COLS, "", 255) == 0) {
                fprintf(stderr, "Error writing the edge image, %s.\n", outfilename);
                exit(1);
            }
            // Record and send current simulation time
           previous_received_Time = received_Time;
           received_Time = sc_time_stamp();
           latency = received_Time - sent_time;
           double Time_diff = (received_Time - previous_received_Time).to_seconds();
           msg = "  | at Time: " + received_Time.to_string() +  " : | Received Frame no.: " +  std::to_string(i) + " | with " + latency.to_string() +" Delay.";
           SC_REPORT_INFO(name(), msg.c_str());
           msg = "  | at Time: " + received_Time.to_string() + " : | " + std::to_string(Time_diff) + " seconds after previous frame, | with: "+ std::to_string((1/Time_diff)) +" FPS.";
           SC_REPORT_INFO(name(), msg.c_str());
        }

        sc_stop();
    }


    int write_pgm_image(const char *outfilename, unsigned char *image, int rows, int cols, const char *comment, int maxval) {
        FILE *fp = fopen(outfilename, "w");
        if (fp == NULL) {
            fprintf(stderr, "Error writing the file %s.\n", outfilename);
            return 0;
        }

        fprintf(fp, "P5\n%d %d\n", cols, rows);
        if (comment != NULL && strlen(comment) <= 70)
            fprintf(fp, "# %s\n", comment);
        fprintf(fp, "%d\n", maxval);

        if ((unsigned)rows != fwrite(image, cols, rows, fp)) {
            fprintf(stderr, "Error writing the image data.\n");
            fclose(fp);
            return 0;
        }

        fclose(fp);
        return 1;
    }

    SC_CTOR(Monitor) {
        SC_THREAD(save_images);
        SET_STACK_SIZE
    }
};

SC_MODULE(DataIn){
   sc_fifo_in<IMAGE> ImgIn; //Input  port connected to FIFO channel
   sc_fifo_out<IMAGE> ImgOut; //Output port connected to FIFO channel
   IMAGE Image;

   void read_data(void){
   while (true)
   {

      ImgIn.read(Image);
      ImgOut.write(Image);
   }
}
   SC_CTOR(DataIn){
      SC_THREAD(read_data);
      SET_STACK_SIZE
   }
};

SC_MODULE(DataOut){
   sc_fifo_in<IMAGE> ImgIn;
   sc_fifo_out<IMAGE> ImgOut;
   IMAGE EdgeImage;

   void write_data(void){
   while (true)
   {
      /****************************************************************************
       * Read a processed image from the input FIFO channel form  the DUT Module
       ****************************************************************************/
      ImgIn.read(EdgeImage);
      /****************************************************************************
       *  Write the processed image to the output FIFO channel to the Monitor Module
       ****************************************************************************/
      ImgOut.write(EdgeImage);
   }
}

   SC_CTOR(DataOut){
      SC_THREAD(write_data);
      SET_STACK_SIZE
   }
};

SC_MODULE(Gaussian_Kernel){
    sc_fifo_in<IMAGE> ImgIn;

   sc_fifo_out<KIMAGE> kernelOut1;
   sc_fifo_out<int> ValueOut1;
   sc_fifo_out<IMAGE> ImgOut; 
   IMAGE Image;
   KIMAGE Kernel_Img;
   int windowsize, Center_Value;
   void GaussianKernel(void){
      while(true)
      {
        ImgIn.read(Image);
        /****************************************************************************
        * Create a 1-dimensional gaussian smoothing kernel.
         ****************************************************************************/
        if (VERBOSE)
            printf("   Computing the gaussian smoothing kernel.\n");
        make_gaussian_kernel(SIGMA, Kernel_Img, &windowsize);
        Center_Value = windowsize / 2;
        wait(0, SC_MS); // Simulated delay for Gaussian_Kernel
        /****************************************************************************
            * Write the Smoothed image to the output FIFO channel
            ****************************************************************************/
        kernelOut1.write(Kernel_Img);
        ValueOut1.write(Center_Value);
        ImgOut.write(Image);

      }
}
   void make_gaussian_kernel(float sigma, float *kernel, int *windowsize){
   int i, center;
   float x, fx, sum = 0.0;

   *windowsize = 1 + 2 * ceil(2.5 * sigma);
   center = (*windowsize) / 2;

   if (VERBOSE)
      printf("      The kernel has %d elements.\n", *windowsize);

   for (i = 0; i < (*windowsize); i++)
   {
      x = (float)(i - center);
      fx = pow(2.71828, -0.5 * x * x / (sigma * sigma)) / (sigma * sqrt(6.2831853));
      kernel[i] = fx;
      sum += fx;
   }

   for (i = 0; i < (*windowsize); i++)
      kernel[i] /= sum;

   if (VERBOSE)
   {
      printf("The filter coefficients are:\n");
      for (i = 0; i < (*windowsize); i++)
         printf("kernel[%d] = %f\n", i, kernel[i]);
   }
}

   SC_CTOR(Gaussian_Kernel){
      SC_THREAD(GaussianKernel);
      SET_STACK_SIZE
   }
};


SC_MODULE(Blur_X) {
    sc_fifo_in<IMAGE> ImgIn; 
    sc_fifo_in<KIMAGE> kernelIn; 
    sc_fifo_in<int> Center_ValueIn;  
    sc_fifo_out<KIMAGE> kernelOut2;
    sc_fifo_out<int> ValueOut2;
    sc_fifo_out<FIMAGE> tempOut;

    IMAGE Image;
    KIMAGE Kernel_Img;
    FIMAGE Temp_Img;
    int Center_Value;
    sc_event e1, e2, e3, e4, data_received; // Synchronization events

    void BlurX_Slice1() {   
       while(true){             //ADD WHILE LOOP & WAIT STMT AFTER DATA RECEIVED
        wait(data_received);
        wait(7.88/4, SC_MS); // Wait for the image data to be ready
        process_blur_x(0, ROWS / 4, COLS);
        e1.notify(SC_ZERO_TIME); // Notify the main thread
    }}

    void BlurX_Slice2() {
        while(true){            //ADD WHILE LOOP & WAIT STMT AFTER DATA RECEIVED
        wait(data_received);
        wait(7.88/4, SC_MS); // Wait for the image data to be ready
        int start = ROWS / 4;
        int end = ROWS / 2;
        process_blur_x(ROWS / 4, ROWS / 2, COLS);
        e2.notify(SC_ZERO_TIME);
    }}

    void BlurX_Slice3() {
          while(true){            //ADD WHILE LOOP & WAIT STMT AFTER DATA RECEIVED
        wait(data_received);
        wait(7.88/4, SC_MS); // Wait for the image data to be ready
        process_blur_x(ROWS / 2, 3 * ROWS / 4, COLS);
        e3.notify(SC_ZERO_TIME);
    }}

    void BlurX_Slice4() {
       while(true){             //ADD WHILE LOOP & WAIT STMT AFTER DATA RECEIVED
        wait(data_received);
        wait(7.88/4, SC_MS); // Wait for the image data to be ready
        process_blur_x(3 * ROWS / 4, ROWS, COLS);
        e4.notify(SC_ZERO_TIME);
    }}

    void BlurX_Main(void) {    //void BlurX_Main(void)
        while (true) {
            ImgIn.read(Image);
            kernelIn.read(Kernel_Img);
            Center_ValueIn.read(Center_Value);

            // Notify all slice threads
            data_received.notify(SC_ZERO_TIME);

            // Wait for all threads to complete
            wait(e1 & e2 & e3 & e4);

            kernelOut2.write(Kernel_Img);
            ValueOut2.write(Center_Value);
            tempOut.write(Temp_Img);
        }
    }

    void process_blur_x(int start_row, int end_row, int cols) {    //MENTION int row, int cols
   int r, c, rr, cc; /* Counter variables. */
   float dot,        /* Dot product summing variable. */
       sum;          /* Sum of the kernel weights variable. */
   int center;
   /****************************************************************************
    * Blur in the x - direction.
    ****************************************************************************/
   if (VERBOSE)
      printf("   Bluring the image in the X-direction.\n");
        for (int r = start_row; r < end_row; r++) {
            for (int c = 0; c < cols; c++) {
                float dot = 0.0, sum = 0.0;
                for (int cc = -center; cc <= center; cc++) {
                    if ((c + cc) >= 0 && (c + cc) < cols) {
                        dot += (float)Image[r * cols + (c + cc)] * Kernel_Img[center + cc];
                        sum += Kernel_Img[center + cc];
                    }
                }
                Temp_Img[r * cols + c] = dot / sum;
            }
        }
    }

    SC_CTOR(Blur_X) {
        SC_THREAD(BlurX_Main);    //ADD SET STACK SIZE FOR EACH
        SET_STACK_SIZE
        SC_THREAD(BlurX_Slice1);
        SET_STACK_SIZE
        SC_THREAD(BlurX_Slice2);
        SET_STACK_SIZE
        SC_THREAD(BlurX_Slice3);
        SET_STACK_SIZE
        SC_THREAD(BlurX_Slice4);
        SET_STACK_SIZE
    }
};


SC_MODULE(Blur_Y) {
    sc_fifo_in<FIMAGE> tempIn;
    sc_fifo_in<KIMAGE> kernelIn;
    sc_fifo_in<int> Center_ValueIn;
    sc_fifo_out<SIMAGE> Smoothed_ImgOut;

    FIMAGE Temp_Img;
    KIMAGE Kernel_Img;
    int Center_Value;
    SIMAGE smoothedim;
    sc_event e1, e2, e3, e4, data_received;

    void BlurY_Slice1() {
      while(true){
        wait(data_received);
        wait(23.31/4, SC_MS);
        process_blur_y(0, COLS / 4, ROWS, COLS);
        e1.notify(SC_ZERO_TIME);
    }}

    void BlurY_Slice2() {
      while(true){
        wait(data_received);
        wait(23.31/4, SC_MS);
        process_blur_y(COLS / 4, COLS / 2, ROWS, COLS);
        e2.notify(SC_ZERO_TIME);
    }}

    void BlurY_Slice3() {
      while(true){
        wait(data_received);
        wait(23.31/4, SC_MS);
        process_blur_y(COLS / 2, COLS / 4, ROWS, COLS);
        e3.notify(SC_ZERO_TIME);
    }}

    void BlurY_Slice4() {
      while(true){
        wait(data_received);
        wait(23.31/4, SC_MS);
        process_blur_y(3 * COLS / 4, COLS, ROWS, COLS);
        e4.notify(SC_ZERO_TIME);
    }}

    void BlurY_Main() {
        while (true) {
            tempIn.read(Temp_Img);
            kernelIn.read(Kernel_Img);
            Center_ValueIn.read(Center_Value);

            data_received.notify(SC_ZERO_TIME);
            wait(e1 & e2 & e3 & e4);

            Smoothed_ImgOut.write(smoothedim);
        }
    }

    void process_blur_y(int start_col, int end_col, int rows, int cols) {    //MENTION int row, int cols
   int r, c, rr, cc; /* Counter variables. */
   float dot,        /* Dot product summing variable. */
       sum;          /* Sum of the kernel weights variable. */
   int center;
   /****************************************************************************
    * Blur in the y - direction.
    ****************************************************************************/
   if (VERBOSE)
      printf("   Bluring the image in the Y-direction.\n");
        for (int c = start_col; c < end_col; c++) {
            for (int r = 0; r < rows; r++) {
                float dot = 0.0, sum = 0.0;
                for (int rr = -center; rr <= center; rr++) {
                    if ((r + rr) >= 0 && (r + rr) < rows) {
                        dot += Temp_Img[(r + rr) * cols + c] * Kernel_Img[center + rr];
                        sum += Kernel_Img[center + rr];
                    }
                }
                smoothedim[r * cols + c] = (short int)(dot * BOOSTBLURFACTOR / sum + 0.5);
            }
        }  
    }

    SC_CTOR(Blur_Y) {
        SC_THREAD(BlurY_Main);
        SET_STACK_SIZE
        SC_THREAD(BlurY_Slice1);
        SET_STACK_SIZE
        SC_THREAD(BlurY_Slice2);
        SET_STACK_SIZE
        SC_THREAD(BlurY_Slice3);
        SET_STACK_SIZE
        SC_THREAD(BlurY_Slice4);
        SET_STACK_SIZE
    }
};


SC_MODULE(Gaussian_Smooth){
   // Instantiate sub-modules
   Gaussian_Kernel gaussian_kernel;
   Blur_X blur_x;
   Blur_Y blur_y;

   sc_fifo_in<IMAGE> ImgIn;
   sc_fifo_out<SIMAGE> Smoothed_ImgOut;

   sc_fifo<IMAGE> img; 
   sc_fifo<KIMAGE> knl_q1, knl_q2; 
   sc_fifo<FIMAGE> tmp_q;          
   sc_fifo<int> int_q1, int_q2;

   sc_fifo<IMAGE> nms;

   void before_end_of_elaboration(void){
      gaussian_kernel.ImgIn.bind(ImgIn);

      gaussian_kernel.ImgOut.bind(img);
      blur_x.ImgIn.bind(img);
      
      gaussian_kernel.kernelOut1.bind(knl_q1);
      blur_x.kernelIn.bind(knl_q1);

      blur_x.kernelOut2.bind(knl_q2);//
      blur_y.kernelIn.bind(knl_q2);

      gaussian_kernel.ValueOut1.bind(int_q1);
      blur_x.Center_ValueIn.bind(int_q1);

      blur_x.ValueOut2.bind(int_q2);//
      blur_y.Center_ValueIn.bind(int_q2);

      blur_x.tempOut.bind(tmp_q);
      blur_y.tempIn.bind(tmp_q);

      blur_y.Smoothed_ImgOut.bind(Smoothed_ImgOut);
   }
   SC_CTOR(Gaussian_Smooth)
       : gaussian_kernel("Gaussian_Kernel"),
         blur_x("Blur_X"),
         blur_y("Blur_y"),
         knl_q1("knl_q1", 1),
         knl_q2("knl_q2", 1),
         tmp_q("tmp_q", 1),
         int_q1("int_q1", 1),
         int_q2("int_q2", 1)
   {
   }
};



SC_MODULE(Derivative_X_Y){
   sc_fifo_in<SIMAGE> Smoothed_In;
   sc_fifo_out<SIMAGE> ImgOutX1, ImgOutY1;
   SIMAGE smoothedim, ImgX, ImgY;

   void Derivative(void){
   while (true)
   {
      Smoothed_In.read(smoothedim);
      /****************************************************************************
       * Compute the first derivative in the x and y directions.
       ****************************************************************************/
      if (VERBOSE)
         printf("Computing the X and Y first derivatives.\n");
      derivative_x_y(smoothedim, ROWS, COLS, ImgX, ImgY);
      wait(16.24, SC_MS);

      /****************************************************************************
       * Write the  Derivatived image of directions (x, y) to the output FIFO channels to the Magnitude and  Non_Maximal_Suppression Module
       ****************************************************************************/
      ImgOutX1.write(ImgX);
      ImgOutY1.write(ImgY);

   }
}

   void derivative_x_y(short int *smoothedim, int rows, int cols,
                                    short int *delta_x, short int *delta_y){
   int r, c, pos;

   /****************************************************************************
    * Compute the x-derivative. Adjust the derivative at the borders to avoid
    * losing pixels.
    ****************************************************************************/
   if (VERBOSE)
      printf("   Computing the X-direction derivative.\n");
   for (r = 0; r < rows; r++)
   {
      pos = r * cols;
      delta_x[pos] = smoothedim[pos + 1] - smoothedim[pos];
      pos++;
      for (c = 1; c < (cols - 1); c++, pos++)
      {
         delta_x[pos] = smoothedim[pos + 1] - smoothedim[pos - 1];
      }
      delta_x[pos] = smoothedim[pos] - smoothedim[pos - 1];
   }

   /****************************************************************************
    * Compute the y-derivative. Adjust the derivative at the borders to avoid
    * losing pixels.
    ****************************************************************************/
   if (VERBOSE)
      printf("   Computing the Y-direction derivative.\n");
   for (c = 0; c < cols; c++)
   {
      pos = c;
      delta_y[pos] = smoothedim[pos + cols] - smoothedim[pos];
      pos += cols;
      for (r = 1; r < (rows - 1); r++, pos += cols)
      {
         delta_y[pos] = smoothedim[pos + cols] - smoothedim[pos - cols];
      }
      delta_y[pos] = smoothedim[pos] - smoothedim[pos - cols];
   }
}
   SC_CTOR(Derivative_X_Y)
   {
      SC_THREAD(Derivative);
      SET_STACK_SIZE
   }
};

SC_MODULE(Magnitude_X_Y){
   sc_fifo_in<SIMAGE> deltaX, deltaY;              // Input  port connected to SIMAGE FIFO channel
   sc_fifo_out<SIMAGE> Magnitude1; // Output port connected to  SIMAGE FIFO channel

   SIMAGE ImgX, ImgY, Magnituded_Img;
   SIMAGE smoothedim;
   sc_fifo_out<SIMAGE> ImgOutX2, ImgOutY2;
   void Magnitude(void){
   while (true)
   {
      deltaX.read(ImgX);
      deltaY.read(ImgY);
      /****************************************************************************
       * Compute the magnitude of the gradient.
       ****************************************************************************/
      if (VERBOSE)
         printf("Computing the magnitude of the gradient.\n");
      magnitude_x_y(ImgX, ImgY, ROWS, COLS, Magnituded_Img);
      wait(3.85, SC_MS);

      /****************************************************************************
       * Write the  Magnituded image  to the output FIFO channels to the Non_Maximal_Suppression and Apply_Hysteresis  Module
       ****************************************************************************/
      Magnitude1.write(Magnituded_Img);
      ImgOutX2.write(ImgX);
      ImgOutY2.write(ImgY);
   }
}

   void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,
                                  short int *magnitude){
   int r, c, pos, sq1, sq2;

   for (r = 0, pos = 0; r < rows; r++)
   {
      for (c = 0; c < cols; c++, pos++)
      {
         sq1 = (int)delta_x[pos] * (int)delta_x[pos];
         sq2 = (int)delta_y[pos] * (int)delta_y[pos];
         magnitude[pos] = (short)(0.5 + sqrt((float)sq1 + (float)sq2));
      }
   }
}
   SC_CTOR(Magnitude_X_Y){
      SC_THREAD(Magnitude);
      SET_STACK_SIZE
   }
};


SC_MODULE(Non_Maximal_Suppression){
   sc_fifo_in<SIMAGE> Mag, deltaX, deltaY; // Input  port connected to SIMAGE FIFO channel
   sc_fifo_out<IMAGE> SuppressedOut;               // Output port connected to  SIMAGE FIFO channel
   sc_fifo_out<SIMAGE> Magnitude2; // Output port connected to  SIMAGE FIFO channel
   SIMAGE Magnituded_Img, ImgX, ImgY;
   IMAGE Suppressed_Img;

   void NonMaximalSuppression(void){
   while (true)
   {
      Mag.read(Magnituded_Img);
      deltaX.read(ImgX);
      deltaY.read(ImgY);
      /****************************************************************************
       * Perform non-maximal suppression.
       ****************************************************************************/
      if (VERBOSE)
         printf("Doing the non-maximal suppression.\n");
      non_max_supp(Magnituded_Img, ImgX, ImgY, ROWS, COLS, Suppressed_Img);
      wait(32.06, SC_MS);

      /****************************************************************************
       * Write the  Suppressed and Magnituded image  to the output FIFO channels to the Apply_Hysteresis  Module
       ****************************************************************************/
      SuppressedOut.write(Suppressed_Img);
      Magnitude2.write(Magnituded_Img);
   }
}

   void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols,
                                           unsigned char *result)
{
   int rowcount, colcount, count;
   short *magrowptr, *magptr;
   short *gxrowptr, *gxptr;
   short *gyrowptr, *gyptr, z1, z2;
   short m00;
   short gx = 0;
   short gy = 0;
   // float mag1, mag2;
   // float xperp = 0;
   // float yperp = 0;
   int mag1, mag2;
   int xperp = 0;
   int yperp = 0;
   unsigned char *resultrowptr, *resultptr;

   /****************************************************************************
    * Zero the edges of the result image.
    ****************************************************************************/
   for (count = 0, resultrowptr = result, resultptr = result + ncols * (nrows - 1);
        count < ncols; resultptr++, resultrowptr++, count++)
   {
      *resultrowptr = *resultptr = (unsigned char)0;
   }

   for (count = 0, resultptr = result, resultrowptr = result + ncols - 1;
        count < nrows; count++, resultptr += ncols, resultrowptr += ncols)
   {
      *resultptr = *resultrowptr = (unsigned char)0;
   }

   /****************************************************************************
    * Suppress non-maximum points.
    ****************************************************************************/
   for (rowcount = 1, magrowptr = mag + ncols + 1, gxrowptr = gradx + ncols + 1,
       gyrowptr = grady + ncols + 1, resultrowptr = result + ncols + 1;
        rowcount < nrows - 1; /* bug fix 10/05/23, RD */
        rowcount++, magrowptr += ncols, gyrowptr += ncols, gxrowptr += ncols,
       resultrowptr += ncols)
   {
      for (colcount = 1, magptr = magrowptr, gxptr = gxrowptr, gyptr = gyrowptr,
          resultptr = resultrowptr;
           colcount < ncols - 1; /* bug fix 10/05/23, RD */
           colcount++, magptr++, gxptr++, gyptr++, resultptr++)
      {
         m00 = *magptr;
         if (m00 == 0)
         {
            *resultptr = (unsigned char)NOEDGE;
         }
         else
         {
            // xperp = -(gx = *gxptr) / ((float)m00);
            // yperp = (gy = *gyptr) / ((float)m00);
	         gx = *gxptr;
            gy = *gyptr;
            xperp = -(gx<<16)/m00;
            yperp = (gy<<16)/m00;
         }

         if (gx >= 0)
         {
            if (gy >= 0)
            {
               if (gx >= gy)
               {
                  /* 111 */
                  /* Left point */
                  z1 = *(magptr - 1);
                  z2 = *(magptr - ncols - 1);

                  mag1 = (m00 - z1) * xperp + (z2 - z1) * yperp;

                  /* Right point */
                  z1 = *(magptr + 1);
                  z2 = *(magptr + ncols + 1);

                  mag2 = (m00 - z1) * xperp + (z2 - z1) * yperp;
               }
               else
               {
                  /* 110 */
                  /* Left point */
                  z1 = *(magptr - ncols);
                  z2 = *(magptr - ncols - 1);

                  mag1 = (z1 - z2) * xperp + (z1 - m00) * yperp;

                  /* Right point */
                  z1 = *(magptr + ncols);
                  z2 = *(magptr + ncols + 1);

                  mag2 = (z1 - z2) * xperp + (z1 - m00) * yperp;
               }
            }
            else
            {
               if (gx >= -gy)
               {
                  /* 101 */
                  /* Left point */
                  z1 = *(magptr - 1);
                  z2 = *(magptr + ncols - 1);

                  mag1 = (m00 - z1) * xperp + (z1 - z2) * yperp;

                  /* Right point */
                  z1 = *(magptr + 1);
                  z2 = *(magptr - ncols + 1);

                  mag2 = (m00 - z1) * xperp + (z1 - z2) * yperp;
               }
               else
               {
                  /* 100 */
                  /* Left point */
                  z1 = *(magptr + ncols);
                  z2 = *(magptr + ncols - 1);

                  mag1 = (z1 - z2) * xperp + (m00 - z1) * yperp;

                  /* Right point */
                  z1 = *(magptr - ncols);
                  z2 = *(magptr - ncols + 1);

                  mag2 = (z1 - z2) * xperp + (m00 - z1) * yperp;
               }
            }
         }
         else
         {
            if ((gy = *gyptr) >= 0)
            {
               if (-gx >= gy)
               {
                  /* 011 */
                  /* Left point */
                  z1 = *(magptr + 1);
                  z2 = *(magptr - ncols + 1);

                  mag1 = (z1 - m00) * xperp + (z2 - z1) * yperp;

                  /* Right point */
                  z1 = *(magptr - 1);
                  z2 = *(magptr + ncols - 1);

                  mag2 = (z1 - m00) * xperp + (z2 - z1) * yperp;
               }
               else
               {
                  /* 010 */
                  /* Left point */
                  z1 = *(magptr - ncols);
                  z2 = *(magptr - ncols + 1);

                  mag1 = (z2 - z1) * xperp + (z1 - m00) * yperp;

                  /* Right point */
                  z1 = *(magptr + ncols);
                  z2 = *(magptr + ncols - 1);

                  mag2 = (z2 - z1) * xperp + (z1 - m00) * yperp;
               }
            }
            else
            {
               if (-gx > -gy)
               {
                  /* 001 */
                  /* Left point */
                  z1 = *(magptr + 1);
                  z2 = *(magptr + ncols + 1);

                  mag1 = (z1 - m00) * xperp + (z1 - z2) * yperp;

                  /* Right point */
                  z1 = *(magptr - 1);
                  z2 = *(magptr - ncols - 1);

                  mag2 = (z1 - m00) * xperp + (z1 - z2) * yperp;
               }
               else
               {
                  /* 000 */
                  /* Left point */
                  z1 = *(magptr + ncols);
                  z2 = *(magptr + ncols + 1);

                  mag1 = (z2 - z1) * xperp + (m00 - z1) * yperp;

                  /* Right point */
                  z1 = *(magptr - ncols);
                  z2 = *(magptr - ncols - 1);

                  mag2 = (z2 - z1) * xperp + (m00 - z1) * yperp;
               }
            }
         }

         if ((mag1 > 0.0) || (mag2 > 0.0))
         {
            *resultptr = (unsigned char)NOEDGE;
         }
         else
         {
            if (mag2 == 0.0)
               *resultptr = (unsigned char)NOEDGE;
            else
               *resultptr = (unsigned char)POSSIBLE_EDGE;
         }
      }
   }
}

   SC_CTOR(Non_Maximal_Suppression){
      SC_THREAD(NonMaximalSuppression);
      SET_STACK_SIZE
   }
};


SC_MODULE(Apply_Hysteresis){
   sc_fifo_in<SIMAGE> Mag; // Input  port connected to SIMAGE FIFO channel
   sc_fifo_in<IMAGE> Suppressed_ImgIn;  // Input  port connected to SIMAGE FIFO channel

   sc_fifo_out<IMAGE> Edge; // Output port connected to  IMAGE FIFO channel

   SIMAGE Magnituded_Img;
   IMAGE Suppressed_Img, Edge_Img;

   void Hysteresis(void){
   while (true)
   {
      /****************************************************************************
       * Read an image from the input FIFO channel form the Magnitude_X_Y and Non_Maximal_Suppression Module
       ****************************************************************************/
      Suppressed_ImgIn.read(Suppressed_Img);
      Mag.read(Magnituded_Img);
      /****************************************************************************
       * Use hysteresis to mark the edge pixels.
       ****************************************************************************/
      if (VERBOSE)
         printf("Doing hysteresis thresholding.\n");
      apply_hysteresis(Magnituded_Img, Suppressed_Img, ROWS, COLS, TLOW, THIGH, Edge_Img);
      wait(24.01, SC_MS);

      /****************************************************************************
       * Write the  Edge image  to the output FIFO channels to the DataOut module
       ****************************************************************************/
      Edge.write(Edge_Img);
   }
}
   void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,
                                        float tlow, float thigh, unsigned char *edge){
   int r, c, pos, numedges, highcount, lowthreshold, highthreshold, hist[32768];
   short int maximum_mag = 0;

   /****************************************************************************
    * Initialize the edge map to possible edges everywhere the non-maximal
    * suppression suggested there could be an edge except for the border. At
    * the border we say there can not be an edge because it makes the
    * follow_edges algorithm more efficient to not worry about tracking an
    * edge off the side of the image.
    ****************************************************************************/
   for (r = 0, pos = 0; r < rows; r++)
   {
      for (c = 0; c < cols; c++, pos++)
      {
         if (nms[pos] == POSSIBLE_EDGE)
            edge[pos] = POSSIBLE_EDGE;
         else
            edge[pos] = NOEDGE;
      }
   }

   for (r = 0, pos = 0; r < rows; r++, pos += cols)
   {
      edge[pos] = NOEDGE;
      edge[pos + cols - 1] = NOEDGE;
   }
   pos = (rows - 1) * cols;
   for (c = 0; c < cols; c++, pos++)
   {
      edge[c] = NOEDGE;
      edge[pos] = NOEDGE;
   }

   /****************************************************************************
    * Compute the histogram of the magnitude image. Then use the histogram to
    * compute hysteresis thresholds.
    ****************************************************************************/
   for (r = 0; r < 32768; r++)
      hist[r] = 0;
   for (r = 0, pos = 0; r < rows; r++)
   {
      for (c = 0; c < cols; c++, pos++)
      {
         if (edge[pos] == POSSIBLE_EDGE)
            hist[mag[pos]]++;
      }
   }

   /****************************************************************************
    * Compute the number of pixels that passed the nonmaximal suppression.
    ****************************************************************************/
   for (r = 1, numedges = 0; r < 32768; r++)
   {
      if (hist[r] != 0)
         maximum_mag = r;
      numedges += hist[r];
   }

   highcount = (int)(numedges * thigh + 0.5);

   /****************************************************************************
    * Compute the high threshold value as the (100 * thigh) percentage point
    * in the magnitude of the gradient histogram of all the pixels that passes
    * non-maximal suppression. Then calculate the low threshold as a fraction
    * of the computed high threshold value. John Canny said in his paper
    * "A Computational Approach to Edge Detection" that "The ratio of the
    * high to low threshold in the implementation is in the range two or three
    * to one." That means that in terms of this implementation, we should
    * choose tlow ~= 0.5 or 0.33333.
    ****************************************************************************/
   r = 1;
   numedges = hist[1];
   while ((r < (maximum_mag - 1)) && (numedges < highcount))
   {
      r++;
      numedges += hist[r];
   }
   highthreshold = r;
   lowthreshold = (int)(highthreshold * tlow + 0.5);

   if (VERBOSE)
   {
      printf("The input low and high fractions of %f and %f computed to\n",
             tlow, thigh);
      printf("magnitude of the gradient threshold values of: %d %d\n",
             lowthreshold, highthreshold);
   }

   /****************************************************************************
    * This loop looks for pixels above the highthreshold to locate edges and
    * then calls follow_edges to continue the edge.
    ****************************************************************************/
   for (r = 0, pos = 0; r < rows; r++)
   {
      for (c = 0; c < cols; c++, pos++)
      {
         if ((edge[pos] == POSSIBLE_EDGE) && (mag[pos] >= highthreshold))
         {
            edge[pos] = EDGE;
            follow_edges((edge + pos), (mag + pos), lowthreshold, cols);
         }
      }
   }

   /****************************************************************************
    * Set all the remaining possible edges to non-edges.
    ****************************************************************************/
   for (r = 0, pos = 0; r < rows; r++)
   {
      for (c = 0; c < cols; c++, pos++)
         if (edge[pos] != EDGE)
            edge[pos] = NOEDGE;
   }
}
   void follow_edges(unsigned char *edgemapptr, short *edgemagptr, short lowval,
                                    int cols)
{
   short *tempmagptr;
   unsigned char *tempmapptr;
   int i;
   int x[8] = {1, 1, 0, -1, -1, -1, 0, 1},
       y[8] = {0, 1, 1, 1, 0, -1, -1, -1};

   for (i = 0; i < 8; i++)
   {
      tempmapptr = edgemapptr - y[i] * cols + x[i];
      tempmagptr = edgemagptr - y[i] * cols + x[i];

      if ((*tempmapptr == POSSIBLE_EDGE) && (*tempmagptr > lowval))
      {
         *tempmapptr = (unsigned char)EDGE;
         follow_edges(tempmapptr, tempmagptr, lowval, cols);
      }
   }
}
   // Constructor: Initializes the SC_THREAD for Hysteresis processing
   SC_CTOR(Apply_Hysteresis)
   {
      SC_THREAD(Hysteresis);
      SET_STACK_SIZE
   }
};


SC_MODULE(DUT){
   // Instantiate sub-modules
   Gaussian_Smooth gaussian_smooth;
   Derivative_X_Y derivative_x_y;
   Magnitude_X_Y magnitude_x_y;
   Non_Maximal_Suppression non_maximal_suppression;
   Apply_Hysteresis apply_hysteresis;

   sc_fifo_in<IMAGE> ImgIn;
   sc_fifo_out<IMAGE> ImgOut;

   sc_fifo<SIMAGE> q1, q2, q3, q4, q5, q6, q7;
   sc_fifo<IMAGE> nms;

   void before_end_of_elaboration(void){
      gaussian_smooth.ImgIn.bind(ImgIn);
      gaussian_smooth.Smoothed_ImgOut.bind(q1);
      derivative_x_y.Smoothed_In.bind(q1);
      derivative_x_y.ImgOutX1.bind(q2);
      magnitude_x_y.deltaX.bind(q2);
      magnitude_x_y.ImgOutX2.bind(q3);
      non_maximal_suppression.deltaX.bind(q3);
      derivative_x_y.ImgOutY1.bind(q4);
      magnitude_x_y.deltaY.bind(q4);
      magnitude_x_y.ImgOutY2.bind(q5);
      non_maximal_suppression.deltaY.bind(q5);
      magnitude_x_y.Magnitude1.bind(q6);
      non_maximal_suppression.Mag.bind(q6);
      non_maximal_suppression.Magnitude2.bind(q7);
      apply_hysteresis.Mag.bind(q7);
      non_maximal_suppression.SuppressedOut.bind(nms);
      apply_hysteresis.Suppressed_ImgIn.bind(nms);
      apply_hysteresis.Edge.bind(ImgOut);
   }

   SC_CTOR(DUT)
       : gaussian_smooth("Gaussian_Smooth"),
         derivative_x_y("Derivative_X_Y"),
         magnitude_x_y("Magnitude_X_Y"),
         non_maximal_suppression("Non_Maximal_Suppression"),
         apply_hysteresis("Apply_Hysteresis"),
         q1("q1", 1),  
         q2("q2", 1),  
         q3("q3", 1),  
         q4("q4", 1),  
         q5("q5", 1),  
         q6("q6", 1),  
         q7("q7", 1),  
         nms("nms", 1) 
   {
   }
};

SC_MODULE(Platform){
   // Instantiate sub-modules
   DataIn data_in;   
   DUT dut;          
   DataOut data_out; 

   sc_fifo_in<IMAGE> ImgIn;   
   sc_fifo_out<IMAGE> ImgOut; 

   sc_fifo<IMAGE> q1; // FIFO channel between DataIn and DUT
   sc_fifo<IMAGE> q2; // FIFO channel between DUT and DataOut

   void before_end_of_elaboration(void){
      data_in.ImgIn.bind(ImgIn);
      data_in.ImgOut.bind(q1);
      dut.ImgIn.bind(q1);

      dut.ImgOut.bind(q2);
      data_out.ImgIn.bind(q2);
      data_out.ImgOut.bind(ImgOut);
   }

   SC_CTOR(Platform)
       : data_in("DataIn"),
         dut("Canny"),
         data_out("DataOut"),
         q1("q1", 1),
         q2("q2", 1)
   {
   }
};


SC_MODULE(Top)
{
   Stimulus stimulus;
   Monitor monitor;
   Platform platform;

   sc_fifo<IMAGE> q1, q2;
   sc_fifo<sc_time> time_fifo;

   void before_end_of_elaboration(void)
   {
      stimulus.ImgOut.bind(q1);
      platform.ImgIn.bind(q1);

      platform.ImgOut.bind(q2);
      monitor.ImgIn(q2);

      stimulus.TimeOut.bind(time_fifo);
      monitor.TimeIn(time_fifo);
   }

   SC_CTOR(Top)
       : stimulus("Stimulus"),
         monitor("Monitor"),
         platform("platform"),
         q1("q1", 1),
         q2("q2", 1),
         time_fifo("time_fifo", 30)
   {
   }
};

Top top("top");

int sc_main(int argc, char *argv[])
{
   sc_report_handler::set_actions(SC_INFO, SC_DISPLAY);
   sc_start();
   return 0;
}



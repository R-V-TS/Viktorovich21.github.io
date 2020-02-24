#include <emscripten.h>
#include "dct_filter.cpp"

extern "C"{
    float *coefficients;
    uint8_t* buffer;
    int image_width = 0, image_height = 0;

    double maximus[16] = {0.943050, 0.091163, 2.87031, 24.574611, 0.33017, 0.045503, 2.7699235, 14.091056, 0.281216, 0.01739, 9.705039, 204.841732, 0.1549, 0.00833551, 18.81166638, 757.186529};
    double minimus[16] = {0.236771, 0.002437, -3.638965, 1.380515, 0.0492139, 0.001816060, -0.491994, 1.966563, 0.006327727, 0.00006727, -0.540810, 1.856624, 0.0014082, 0.0000066458, 0.187976, 2.472991};

    EMSCRIPTEN_KEEPALIVE
    uint8_t* buffer_init(int width, int height){
        buffer = new uint8_t[width*height];
        image_width = width;
        image_height = height;
        return buffer;
    }

    void getImageBlockUInt(uint8_t* image, int i_, int j_, int image_width, int window_size, float* block) {
        for (int i = i_; i < window_size+i_; i++)
            for (int j = j_; j < (window_size)+j_; j++) {
                block[window_size * (i-i_) + (j - j_)] = ((float) image[(image_width * (i)) + (j)]);
            }
    }

    void getImageBlockUInt2UInt(uint8_t* image, int i_, int j_, int image_width, int window_size, uint8_t* block) {
        for (int i = i_; i < window_size+i_; i++)
            for (int j = j_; j < (window_size)+j_; j++) {
                block[window_size * (i-i_) + (j - j_)] = image[(image_width * (i)) + (j)];
            }
    }

    void getImageBlockF(float* image, int i_, int j_, int image_width, int window_size, float* block) {
        for (int i = i_; i < window_size+i_; i++)
            for (int j = j_; j < (window_size)+j_; j++) {
                block[window_size * (i-i_) + (j - j_)] = image[(image_width * (i)) + (j)];
            }
    }

    EMSCRIPTEN_KEEPALIVE
    uint8_t* createImageBuffer(int length){
        return (uint8_t*)malloc(length*sizeof(uint8_t));
    }

    EMSCRIPTEN_KEEPALIVE
    float* getDCTOffset(int length){
        return (float*)malloc(length*sizeof(float));
    }

    EMSCRIPTEN_KEEPALIVE
    void DCT_image(uint8_t* image_, int width_, int height_, int wind_size_, float* im_temp)
    {
        float *DCT_creator_mtx;
        float *DCT_creator_mtx_T;
        switch (wind_size_) {
            case 2:
                DCT_creator_mtx = &DCT_Creator2[0][0];
                DCT_creator_mtx_T = &DCT_Creator2_T[0][0];
                break;
            case 4:
                DCT_creator_mtx = &DCT_Creator4[0][0];
                DCT_creator_mtx_T = &DCT_Creator4_T[0][0];
                break;
            case 8:
                DCT_creator_mtx = &DCT_Creator8[0][0];
                DCT_creator_mtx_T = &DCT_Creator8_T[0][0];
                break;
            case 16:
                DCT_creator_mtx = &DCT_Creator16[0][0];
                DCT_creator_mtx_T = &DCT_Creator16_T[0][0];
                break;
            case 32:
                DCT_creator_mtx = &DCT_Creator32[0][0];
                DCT_creator_mtx_T = &DCT_Creator32_T[0][0];
                break;
        }
        float *block = new float[wind_size_*wind_size_];
        float *temp = new float[wind_size_*wind_size_];

        for (int i = 0; i < height_; i += wind_size_) {
            for (int j = 0; j < width_; j += wind_size_) {
                getImageBlockUInt(image_, i, j, width_, wind_size_, block); // get block from image
                MultiplyMatrix(DCT_creator_mtx, block, temp, wind_size_);
                MultiplyMatrix(temp, DCT_creator_mtx_T, block, wind_size_);   // D*A*D'

                for (int l = i; l < wind_size_ + i; l++) // Put DCT block to dct temp block
                    for (int t = j; t < wind_size_ + (j); t++) {
                        *(im_temp + (width_ * l) + t) = *(block + (wind_size_ * (l - i)) + (t - j));
                    }
            }
        }
    }

    EMSCRIPTEN_KEEPALIVE
    float* getCoffOffset(){
        coefficients = new float[16];
        return coefficients;
    }

    EMSCRIPTEN_KEEPALIVE
    void calcDCTCoefficients() {
        int window_size = 8;

        int matrix_flag[8][8] = {
                {0,  1,  1,  1,  1,  2,  2,  2},
                {1,  1,  1,  1,  2,  2,  2,  3},
                {1,  1,  1,  2,  2,  2,  3,  3},
                {1,  1,  2,  2,  2,  3,  3,  3},
                {1,  2,  2,  2,  3,  3,  3,  4},
                {2,  2,  2,  3,  3,  3,  4,  4},
                {2,  2,  3,  3,  3,  4,  4,  4},
                {2,  3,  3,  3,  4,  4,  4,  4}
        };

        float* EST = new float[(image_width/8)*(image_height/8)*4];
        float* block = new float[window_size*window_size];
        float* DCT_ARRAY = new float[image_height*image_width]; 
        DCT_image(buffer, image_width, image_height, window_size, DCT_ARRAY);
        int z = 0;

        double slices_sum[4] = {0,0,0,0};
        double mean = 0;
        double variance = 0;
        double skeweness = 0;
        double kurtosis = 0;
        int count = 0;
        double fulldct = 0;
        int pixel_p = 0;
        for(int i = 0; i < image_height; i+=window_size)
        {
            for(int j = 0; j < image_width; j+=window_size)
            {
                fulldct = 0;
                getImageBlockF(DCT_ARRAY, i, j, image_width, window_size, block);
                block[0] = 0;

                for(int l = 0; l < window_size; l++) {
                    for (int k = 0; k < window_size; k++) {
                        block[(l*window_size) + k] = block[(l*window_size) + k] * block[(l*window_size) + k];
                        fulldct += block[(l * window_size) + k];
                        slices_sum[matrix_flag[l][k]-1] += block[(l * window_size) + k];
                    }
                }

                for(int l = 0; l < 4; l++) {
                    slices_sum[l] /= fulldct;
                    EST[pixel_p] = slices_sum[l];
                    pixel_p++;
                    slices_sum[l] = 0;
                }
            }
        }

        for(int l = 0; l < 4; l++)
        {
            mean = 0;
            pixel_p = l;
            for(int i = 0; i < image_height/8; i++)
            {
                for(int j = 0; j < image_width/8; j++)
                {
                    mean += EST[pixel_p];
                    pixel_p += 4;
                    count++;
                }
            }
            count = image_width*image_height;
            mean /= count;

            pixel_p = l;
            variance = 0;
            skeweness = 0;
            kurtosis = 0;
            for(int i = 0; i < image_height/8; i++)
            {
                for(int j = 0; j < image_width/8; j++)
                {
                    variance += pow((EST[pixel_p] - mean), 2);
                    skeweness += pow((EST[pixel_p]-mean), 3);
                    kurtosis += pow((EST[pixel_p]-mean), 4);
                    pixel_p += 4;
                }
            }

            skeweness = (skeweness/count)/pow(sqrt(variance/count), 3);
            kurtosis = (kurtosis/count)/pow(variance/count, 2);
            variance = variance/(count-1);
            coefficients[z] = (((1 - (-1)) * ((mean) - minimus[z])) / (maximus[z] - minimus[z])) + (-1);
            coefficients[z+1] = (((1 - (-1)) * ((variance) - minimus[z+1])) / (maximus[z+1] - minimus[z+1])) + (-1);
            coefficients[z+2] = (((1 - (-1)) * ((skeweness) - minimus[z+2])) / (maximus[z+2] - minimus[z+2])) + (-1);
            coefficients[z+3] = (((1 - (-1)) * ((kurtosis) - minimus[z+3])) / (maximus[z+3] - minimus[z+3])) + (-1);
            z += 4;
        }
        free(EST);
    }

    EMSCRIPTEN_KEEPALIVE
    void destroyAll(){
        free(coefficients);
        free(buffer);
    }
        
}
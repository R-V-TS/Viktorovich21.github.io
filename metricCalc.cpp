#include <math.h>
#include <emscripten.h>
#include "calculateDCTFCof.cpp"


extern "C"{
    uint8_t* P_image = nullptr;
    uint8_t* Q_image = nullptr;
    int width, height;

    EMSCRIPTEN_KEEPALIVE
    uint8_t* P_image_init(int width_, int height_){
        P_image = new uint8_t[width_*height_];
        width = width_;
        height = height_;
        return P_image;
    }

    EMSCRIPTEN_KEEPALIVE
    uint8_t* Q_image_init(int width_, int height_){
        Q_image = new uint8_t[width_*height_];
        width = width_;
        height = height_;
        return Q_image;
    }

    EMSCRIPTEN_KEEPALIVE
    void destroyAll_Metrix(){
        free(P_image);
        free(Q_image);
    }

    float CSFCof[8][8] = {
            {1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887},
            {2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911},
            {1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555},
            {1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082},
            {1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222},
            {1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729},
            {0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803},
            {0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950}
    };

    float MaskCof[8][8] = {
            {0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874},
            {0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058},
            {0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888},
            {0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015},
            {0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866},
            {0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815},
            {0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803},
            {0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203}
    };

    float mean(float* im_block, int length){
        float res = 0;
        for(int i = 0; i < length; i++)
            res += im_block[i];
        return res/length;
    }

    float variance(float* im_block, int length)
    {
        float res = 0;
        float mean_im = mean(im_block, length);
        for(int i = 0; i < length; i++){
            res += pow(im_block[i] = mean_im, 2);
        }
        return res/(length-1);
    }

    float meanI(uint8_t* im_block, int length){
        float res = 0;
        for(int i = 0; i < length; i++)
            res += im_block[i];
        return res/length;
    }

    float varianceI(uint8_t* im_block, int length)
    {
        float res = 0;
        float mean_im = meanI(im_block, length);
        for(int i = 0; i < length; i++){
            res += pow(im_block[i] = mean_im, 2);
        }
        return res/(length-1);
    }

    EMSCRIPTEN_KEEPALIVE
    float MSE(){
        float MSE_res;
        float MSE_local = 0;
        for(int i = 0; i < height; i++)
            for(int j = 0; j < width; j++)
            {
                MSE_local += pow(P_image[(i*width) + j] - Q_image[(i*width)+j], 2);
            }
        MSE_local /= (width*height);
        MSE_res = MSE_local;
        return MSE_res;
    }

    EMSCRIPTEN_KEEPALIVE
    float PSNR(){
        float PSNR_res;
        float MSE_im = MSE();
        uint8_t max = 0;
        for(int i = 0; i < height; i++)
            for(int j = 0; j < width; j++)
            {
                if(max < P_image[(i*width)+j]) max = P_image[(i*width)+j];
            }
        PSNR_res = 20*log10f(max/MSE_im);
        return PSNR_res;
    }

    float maskeff(uint8_t* image, float* DCT_im, int window_size)
    {
        float res = 0;
        float pop;
        for(int i = 0; i < window_size; i++)
            for(int j = 0; j < window_size; j++)
            {
                if(i != 0 && j != 0)
                    res += ((DCT_im[(i*window_size)+j]*DCT_im[(i*window_size)+j]) * MaskCof[i][j]);
            }

        pop = varianceI(image, window_size*window_size) * window_size*window_size;
        if(pop != 0){
            float* block;
            getImageBlockUInt(image, 0, 0, window_size, 4, block);
            float pop_1 = variance(block, 4*4) * 16;
            getImageBlockUInt(image, 0, 4, window_size, 4, block);
            pop_1 += variance(block, 4*4) * 16;
            getImageBlockUInt(image, 4,4,window_size,4, block);
            pop_1 += variance(block, 4*4) * 16;
            getImageBlockUInt(image, 4,0,window_size,4, block);
            pop_1 += variance(block, 4*4) * 16;
            pop = pop_1/pop;
        }
        res = sqrt(pop*res)/32;
        return res;
    }

    EMSCRIPTEN_KEEPALIVE
    float PSNRHVSM(){
        float PSNR_HVS_M = 0.f;
        int window_size = 8;
        uint8_t* blockP = new uint8_t[window_size*window_size];
        uint8_t* blockQ = new uint8_t[window_size*window_size];
        float* DCT_blockP;
        float* DCT_blockQ;
        float MaskP, MaskQ;
        float u; // for calculate abs
        int NUM = 0;

        for(int i = 0; i < height; i+= window_size){
            for(int j = 0; j < width; j+= window_size){
                getImageBlockUInt2UInt(P_image, i, j, width, window_size, blockP);
                getImageBlockUInt2UInt(Q_image, i, j, width, window_size, blockQ);
                DCT_blockP = DCT_image(blockP, window_size, window_size, window_size);
                DCT_blockQ = DCT_image(blockQ, window_size, window_size, window_size);
                MaskP = maskeff(blockP, DCT_blockP, window_size);
                MaskQ = maskeff(blockQ, DCT_blockQ, window_size);
                if(MaskQ > MaskP) MaskP = MaskQ;

                for(int k = 0; k < window_size; k++)
                {
                    for(int l = 0; l < window_size; l++)
                    {
                        u = abs(DCT_blockP[(k*window_size)+l] - DCT_blockQ[(k*window_size)+l]);
                        if(k != 0 || l != 0)
                        {
                            if(u < MaskP/MaskCof[k][l]) u = 0;
                            else u -= MaskP/MaskCof[k][l];
                        }
                        PSNR_HVS_M += pow(u*CSFCof[k][l], 2);   // PSNR-HVS-M calculate
                        NUM++;
                    }
                }
            }
        }

        if(NUM != 0)
        {
            PSNR_HVS_M /= NUM;

            if(PSNR_HVS_M == 0) PSNR_HVS_M = 100000;
            else PSNR_HVS_M = 10*log10f(255*255/PSNR_HVS_M);
        }

        delete[] blockP;
        delete[] blockQ;
        return PSNR_HVS_M;
    }
}
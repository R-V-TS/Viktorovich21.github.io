#include <cstdint>
#include <random>
#include <emscripten.h>

using namespace std;

uint8_t* buffer = nullptr;
int im_width = 0, im_height = 0;

extern "C" {
    
    EMSCRIPTEN_KEEPALIVE
    bool buffer_init(int width, int height, int channel){
        buffer = (uint8_t*)malloc(channel*width*height*sizeof(uint8_t));
        im_width = width;
        im_height = height;
        return true;
    }

    EMSCRIPTEN_KEEPALIVE
    bool setPixel(int x, uint8_t intensivity){
        if(x < im_width * 3 * im_height && x >= 0 && buffer != nullptr){
            buffer[x] = intensivity;
            return true;
        }
        return false;
    }

    EMSCRIPTEN_KEEPALIVE
    void distordImageAWGN(float sigma, float mu){
        std::default_random_engine generator; //Random generator engine
        std::normal_distribution<float> distribution(mu, sigma); //distribution vector
        float rand_gen;

        for(int i = 0; i < im_width*im_height*3; i++)
        {
            rand_gen = distribution(generator);
            if(buffer[i] + rand_gen > 255)
                buffer[i] = 255;
            else if(buffer[i] + rand_gen < 0)
                buffer[i] = 0;
            else
                buffer[i] += rand_gen;
        }
    }

    EMSCRIPTEN_KEEPALIVE
    int getPixel(int x){
        if(x < im_width * 3 * im_height && x >= 0 && buffer != nullptr){
            return buffer[x];
        }
        return 0;
    }

    EMSCRIPTEN_KEEPALIVE
    void destroy_buffer(){
        free(buffer);
    }
}
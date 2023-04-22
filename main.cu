#include "stream.cuh"


static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}



int main(int argc, char *argv[])
{
    printf("CUDA Stream\n");

    char filepath[] = "./fhy_direct.bin";
    int data_per_frame = ChirpSize * SampleSize * numRx * 2;
    int byte_per_frame = data_per_frame * sizeof(short);
    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL)
    {
        printf("Cannot open %s file \n", &(filepath[2]));
        return -1;
    }

    short *read_data = (short *)malloc(byte_per_frame);
    int size = 0;
    double cudaDist[FrameSize];
    int frameCnt = 0;

    size = (int)fread(read_data, sizeof(short), data_per_frame, fp);
    frameCnt++;
    // complex paired data is twice less than input size
    Complex_t *base_frame_reshaped = (Complex_t *)malloc(size * sizeof(Complex_t) / 2);
    Complex_t *base_frame_rx0 = (Complex_t *)malloc(ChirpSize * SampleSize * sizeof(Complex_t));

    preProcessing_host(read_data, base_frame_reshaped, size);
    memmove(base_frame_rx0, base_frame_reshaped, ChirpSize * SampleSize * sizeof(Complex_t));

    int rx0_extended_size = nextPow2(ChirpSize * SampleSize);
    Complex_t *frame_reshaped = (Complex_t *)malloc(size * sizeof(Complex_t) / 2);
    Complex_t *frame_reshaped_rx0 = (Complex_t *)malloc(sizeof(Complex_t) * rx0_extended_size);
    /**
     * Allocate device memory
     */
    Complex_t *frame_reshaped_device;
    Complex_t *frame_reshaped_rx0_device;
    Complex_t *base_frame_rx0_device;
    Complex_t *base_frame_device;
    Complex_t *frame_buffer_device; 

    cudaCheckError(cudaMalloc((void **)&frame_buffer_device, sizeof(Complex_t) * size / 2));
    cudaCheckError(cudaMalloc((void **)&frame_reshaped_device, sizeof(Complex_t) * size / 2));
    cudaCheckError(cudaMalloc((void **)&base_frame_device, sizeof(Complex_t) * size / 2));

    cudaCheckError(cudaMalloc((void **)&frame_reshaped_rx0_device, sizeof(Complex_t) * rx0_extended_size));
    cudaCheckError(cudaMalloc((void **)&base_frame_rx0_device, sizeof(Complex_t) * ChirpSize * SampleSize));
    // copy base frame rx0 data from Host to Device
    cudaCheckError(cudaMemcpy(base_frame_rx0_device, base_frame_rx0, ChirpSize * SampleSize * sizeof(Complex_t), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(base_frame_device, base_frame_reshaped,size * sizeof(Complex_t) / 2, cudaMemcpyHostToDevice));

    Timer timer;
    double start = timer.elapsed();
    double fftTime = 0, preProcessingTime = 0, findMaxTime = 0, totalTime = 0, angleTime=0, distTime = 0;
    if ((size = (int)fread(read_data, sizeof(short), data_per_frame, fp)) > 0)
    {

        // cudaDist[frameCnt] = cudaAcceleration(fftTime, preProcessingTime, findMaxTime, totalTime, read_data, base_frame_rx0_device, frame_buffer_device, frame_reshaped_device, frame_reshaped_rx0_device, size, rx0_extended_size);
        cudaDist[frameCnt] = cudaAcceleration(angleTime, distTime,fftTime, preProcessingTime, findMaxTime, totalTime, read_data, base_frame_device, frame_buffer_device, frame_reshaped_device, frame_reshaped_rx0_device, size, rx0_extended_size);
        
        printf("cudaDist[%d] %.6f m\n", frameCnt, cudaDist[frameCnt]);
        
        
        
        frameCnt++;
    }

    double duration = (timer.elapsed() - start);
    printf("CUDA Stream outer total time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * duration, 1000.0 *duration / frameCnt ,frameCnt / duration);
    printf("CUDA Stream inner total time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * totalTime, 1000.0 *totalTime / frameCnt,frameCnt / totalTime);
    printf("CUDA Stream total fft time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * fftTime, 1000.0 * fftTime / frameCnt ,frameCnt / fftTime);
    printf("CUDA Stream total preProcessing time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * preProcessingTime, 1000.0 *preProcessingTime / frameCnt ,frameCnt / preProcessingTime);
    printf("CUDA Stream total findMaxTime time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * findMaxTime, 1000.0 *findMaxTime / frameCnt ,frameCnt / findMaxTime);
    printf("CUDA Stream total distance calculation time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * distTime, 1000.0 *distTime / frameCnt ,frameCnt / distTime);
    printf("CUDA Stream total angle calculation time time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * angleTime, 1000.0 *angleTime / frameCnt ,frameCnt / angleTime);

    // end region
    cudaCheckError(cudaFree(base_frame_device));
    cudaCheckError(cudaFree(frame_buffer_device));
    cudaCheckError(cudaFree(frame_reshaped_device));
    cudaCheckError(cudaFree(frame_reshaped_rx0_device));
    cudaCheckError(cudaFree(base_frame_rx0_device));

    free(base_frame_reshaped);
    free(base_frame_rx0);
    free(frame_reshaped);
    free(frame_reshaped_rx0);
    free(read_data);
    fclose(fp);

    return 0;
}

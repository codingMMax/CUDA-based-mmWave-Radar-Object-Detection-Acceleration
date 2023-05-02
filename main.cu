#include "stream.cuh"

int main(int argc, char *argv[])
{
    for (int i = 0; i < 1; i++)
    {
        printf("CUDA Stream %d\n", i);

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
        double cudaSpeed[FrameSize];
        double cudaAngle[FrameSize];

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
        Complex_t *base_frame_rx0_device;
        Complex_t *base_frame_device;

        cudaCheckError(cudaMalloc((void **)&frame_reshaped_device, sizeof(Complex_t) * size / 2));
        cudaCheckError(cudaMalloc((void **)&base_frame_device, sizeof(Complex_t) * size / 2));

        cudaCheckError(cudaMalloc((void **)&base_frame_rx0_device, sizeof(Complex_t) * ChirpSize * SampleSize));

        long memBufferSize = sizeof(Complex_t) * size + sizeof(Complex_t) * ChirpSize * SampleSize;
        // printf("allocated buffer size %.5f KiBytes \n", (double)(memBufferSize / 1024));

        // copy base frame rx0 data from Host to Device
        cudaCheckError(cudaMemcpy(base_frame_rx0_device, base_frame_rx0, ChirpSize * SampleSize * sizeof(Complex_t), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(base_frame_device, base_frame_reshaped, size * sizeof(Complex_t) / 2, cudaMemcpyHostToDevice));

        /**
         * allocate the memory buffer for distance processing
         */
        Complex_t *rx0_fft_input_device_dist;
        cudaCheckError(cudaMalloc((void **)&rx0_fft_input_device_dist, sizeof(Complex_t) * rx0_extended_size));
        /**
         * allocate memory buffer for angle processing
         */
        Complex_t *rx_fft_input_device_angle;
        Complex_t *frame_reshaped_device_angle;
        Complex_t *base_frame_device_angle;
        cudaCheckError(cudaMalloc((void **)&rx_fft_input_device_angle, sizeof(Complex_t) * rx0_extended_size * (RxSize - 1)));
        cudaCheckError(cudaMalloc((void **)&base_frame_device_angle, sizeof(Complex_t) * SampleSize * ChirpSize * RxSize));
        cudaCheckError(cudaMalloc((void **)&frame_reshaped_device_angle, sizeof(Complex_t) * SampleSize * ChirpSize * RxSize));
        Complex_t *angle_weights_device;
        Complex_t *rx0_fft_input_device_angle;
        Complex_t *angle_matrix_device;
        Complex_t *angle_matrix_res_device;
        int angle_sample_num = 180 / 0.1 + 1;
        cudaCheckError(cudaMalloc((void **)&angle_weights_device, sizeof(Complex_t) * RxSize));
        cudaCheckError(cudaMalloc((void **)&rx0_fft_input_device_angle, sizeof(Complex_t) * rx0_extended_size));
        cudaCheckError(cudaMalloc((void **)&angle_matrix_device, sizeof(Complex_t) * RxSize * angle_sample_num));
        cudaCheckError(cudaMalloc((void **)&angle_matrix_res_device, sizeof(Complex_t) * angle_sample_num));
        /**
         * allocate memory buffer for speed processing
         */
        Complex_t *rx0_extended_fft_input_device;
        Complex_t *rx0_extended_fftRes_transpose;
        int extended_sample_size = nextPow2(SampleSize);
        cudaCheckError(cudaMalloc((void **)&rx0_extended_fftRes_transpose, sizeof(Complex_t) * extended_sample_size * ChirpSize));

        cudaCheckError(cudaMalloc((void **)&rx0_extended_fft_input_device, sizeof(Complex_t) * extended_sample_size * ChirpSize));

        Timer timer;
        double fftTime = 0, speedTime = 0, preProcessingTime = 0, findMaxTime = 0, totalTime = 0, angleTime = 0, distTime = 0;
        double distance, speed, angle;
        bool singleStream = false;
        while ((size = (int)fread(read_data, sizeof(short), data_per_frame, fp)) > 0)
        {

            // cudaAcceleration(singleStream, speed, angle, distance, speedTime, angleTime, distTime,
            //                  fftTime, preProcessingTime, findMaxTime, totalTime,
            //                  read_data, base_frame_device, frame_reshaped_device,
            //                  size, rx0_extended_size);

            cudaMultiStreamAcceleration(read_data, base_frame_device, frame_reshaped_device, rx0_fft_input_device_dist,
                                        rx_fft_input_device_angle, frame_reshaped_device_angle, base_frame_device_angle,
                                        angle_weights_device, rx0_fft_input_device_angle, angle_matrix_device,
                                        angle_matrix_res_device, rx0_extended_fftRes_transpose, rx0_extended_fft_input_device,
                                        size, rx0_extended_size);

            // cudaDist[frameCnt] = distance;
            // cudaAngle[frameCnt] = angle;
            // cudaSpeed[frameCnt] = speed;

            // printf("cudaDist[%d] %.6f m\n", frameCnt, cudaDist[frameCnt]);
            // printf("cudaAngle[%d] %.6f degree\n", frameCnt, cudaAngle[frameCnt]);
            // printf("cudaSpeed[%d] %.6f m/s\n", frameCnt, cudaSpeed[frameCnt]);

            frameCnt++;
        }
        if (singleStream)
        {
            printf("CUDA Stream inner total time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * totalTime, 1000.0 * totalTime / frameCnt, frameCnt / totalTime);
            printf("CUDA Stream total fft time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * fftTime, 1000.0 * fftTime / frameCnt, frameCnt / fftTime);
            printf("CUDA Stream total preProcessing time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * preProcessingTime, 1000.0 * preProcessingTime / frameCnt, frameCnt / preProcessingTime);
            printf("CUDA Stream total findMaxTime time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * findMaxTime, 1000.0 * findMaxTime / frameCnt, frameCnt / findMaxTime);
            printf("CUDA Stream total distance calculation time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * distTime, 1000.0 * distTime / frameCnt, frameCnt / distTime);
            printf("CUDA Stream total angle calculation time time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * angleTime, 1000.0 * angleTime / frameCnt, frameCnt / angleTime);
            printf("CUDA Stream total speed calculation time time %.3f ms, %.3f ms per frame FPS %.3f \n", 1000.0 * speedTime, 1000.0 * speedTime / frameCnt, frameCnt / speedTime);
        } else {

        }
        // end region
        cudaCheckError(cudaFree(base_frame_device));
        cudaCheckError(cudaFree(frame_reshaped_device));
        cudaCheckError(cudaFree(base_frame_rx0_device));
        // distance buffer free
        cudaCheckError(cudaFree(rx0_fft_input_device_dist));
        // angle buffer free
        cudaCheckError(cudaFree(rx_fft_input_device_angle));
        cudaCheckError(cudaFree(frame_reshaped_device_angle));
        cudaCheckError(cudaFree(base_frame_device_angle));
        cudaCheckError(cudaFree(angle_weights_device));
        cudaCheckError(cudaFree(rx0_fft_input_device_angle));
        cudaCheckError(cudaFree(angle_matrix_device));
        cudaCheckError(cudaFree(angle_matrix_res_device));
        // speed buffer free
        cudaCheckError(cudaFree(rx0_extended_fft_input_device));
        cudaCheckError(cudaFree(rx0_extended_fftRes_transpose));

        free(base_frame_reshaped);
        free(base_frame_rx0);
        free(frame_reshaped);
        free(frame_reshaped_rx0);
        free(read_data);
        fclose(fp);
    }

    return 0;
}

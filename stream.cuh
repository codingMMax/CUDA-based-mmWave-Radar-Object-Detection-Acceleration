/**
 * using CUDA stream to pipeline processing stags
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = false)
{
    if (code != cudaSuccess)
    {

        (cudaGetErrorString(code), file, line);
        abort = true;
    }
    if (abort)
    {
        printf("CUDA Error code: %d at %s:%d\n", code, file, line);
        exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

#define SampleSize 100 // the sample number in a chirp, suggesting it should be the power of 2
#define ChirpSize 128  // the chirp number in a frame, suggesting it should be the the power of 2
#define FrameSize 90   // the frame number
#define RxSize 4       // the rx size, which is usually 4
#define numTx 1
#define numRx 4 // the rx size, which is usually 4
#define THREADS_PER_BLOCK 512
#define PI 3.14159265359 // pi
#define fs 2.0e6         // sampling frequency
#define lightSpeed 3.0e08
#define mu 5.987e12 // FM slope
#define f0 77e9
#define lamda lightSpeed / f0
#define d = 0.5 * lamda

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const
    {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
};

struct Complex_t
{
    double real, imag;
};

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

void preProcessing_host(short *OriginalArray, Complex_t *Reshape, int size);
void printComplexCUDA(Complex_t *input, int start, int end, int size);
void printShortCUDA(short *input, int start, int end, int size);
int findAbsMax(Complex_t *ptr, int size);
__device__ int cudaFindAbsMax(Complex_t *ptr, int size);
__device__ static Complex_t cudaComplexSub(Complex_t a, Complex_t b);
__device__ static Complex_t cudaComplexAdd(Complex_t a, Complex_t b);
__device__ static Complex_t cudaComplexMul(Complex_t a, Complex_t b);
__device__ static Complex_t makeComplex(double img, double real);
__device__ static double cudaComplexMol(Complex_t a);
/**
 * device function convert short data into paired complex number
 * @param: input: short type input array with length 'size'.
 * @param: buf: complex type output array with length 'size/2';
 * @param: size: int type indicates the input arary length.
 */
__global__ void short2complex_kernel(short *input, Complex_t *buf, int size);
/**
 * kernel function reshape the input complex array into specified array layout
 * reshape the input complex buffer into specified format
 * input format: chirp0[rx0[sample0-100], rx1[sample0-100], rx2[sample0-100]..] chirp1[rx0[..]]
 * output format: rx0[chirp0[sample0-100],chirp1[0-100]...]
 * @param: destArray: destination array hold the required array layout.
 * @param: srcArray: src array with original array layout.
 * @param: size: int variable indicates the array length.
 */
__global__ void complexReshape_kernel(Complex_t *destArray, Complex_t *srcArray, int size);
/**
 * kernel function extend the rx0 data and fill in the fft input
 * @param: baseFrame:  input base frame with length 'SampleSize * ChirpSize'
 * @param: extendedBuffer: input buffer to store the extended data with length 'rx0_extended_size'.
 * @param: reshaped_frame: already reshaped frame data, with length 'SampleSize * ChiprSize * numRx'.
 * @param: rx0_extended_size: extended rx0 size = nextPow2(SampleSize * ChirpSize)
 *
 */
__global__ void rxExtension_kernel(Complex_t *baseFrame, Complex_t *extendedBuffer, Complex_t *reshaped_frame, int oldSize, int extendedSize);
/**
 * compute the reverse decimal number of input 'num' for given 'bits'
 */
__device__ int bitsReverse(int num, int bits);
/**
 * Kernel function perform bit reverse and element swap for later fft
 * @param: input: 'input' complex input array, with length 'size'.
 * @param: input: 'size' array length.
 * @param: input: 'pow'  power of the input size = log2(size).
 */
__global__ void bitReverseSwap_kernel(Complex_t *input, int size, int pow);

/**
 * kernel function perform butterfly computation fft for input data.
 * @param: input: 'data' complex input array with length 'size'.
 * @param: input: 'size' array length.
 * @param: input: 'stage' int number indicates the current stage for butterfly computation.
 * @param: input: 'pow' power of the input length = log2(size).
 */
__global__ void butterflyFFT_kernel(Complex_t *data, int size, int stage, int pow);

/**
 * Kernel function that initialize the angle weights in 3 stages.
 * Stage1: initialize the 'angle_fft_buffer[i]' = 'reshaped_frame[i] - base_frame[i]'
 * Stage2: apply FFT for the angle_fft_buffer
 * Stage3: assign weight[1,2,3] = fft_res[maxAngleIdx] / extended_size
 */
__global__ void angleWeightInit_kernel(Complex_t *weights, Complex_t *rx0_fft_input_device, Complex_t *rx_fft_res, int maxAngleIdx, int size);
/**
 * Kernel function to initialize the angle matrix
 * @param matrix: input matrix with dimension: RxSize * AngleSampleNum = 'size'.
 * row = RxSize
 * Col = AngleSampleNum
 */
__global__ void angleMatrixInit_kernel(Complex_t *matrix, int size);
/**
 * angle matrix angle weights multiplication kernel
 * @param angle_matrix input matrix with dim: RxSize * num_angle_sample
 * @param angle_weight input matrix with dim: 1 * RxSize
 * @param res output matrix with dim: 1 * num_angle_sample
 */
__global__ void angleMatrixMul_kernel(Complex_t *angle_matrix, Complex_t *angle_weight, Complex_t *res, int num_angle_sample);
/**
 * kernel function for rx0 data padding
 * @param rx0_extended: extended rx0 data with length 'extended_sample_size * ChirpSize'.
 * @param rx0_non_extended: non-extended rx0 data with length 'ChirpSize * SampleSize'.
 * @param base_frame_rx0: rx0 data of base frame with length 'ChirpSize * SampleSize'.
 * @param extended_size: toal length of extended size = 'extended_sample_size * ChirpSize'.
 * @param non_extended_size: total length of non-extended size = 'SampleSize * ChirpSize'.
 * @param extended_sample_size: lenght of extende sample size = 'nexPow2(SampleSize)'.
 */
__global__ void rx0ChirpPadding_kernel(Complex_t *rx0_extended, Complex_t *rx0_non_extended, Complex_t *base_frame_rx0, int extended_size, int non_extended_size, int extended_sample_size);
/**
 * Kernel function to transpose the input matrix
 * @param matrix: input matrix with dim1 x dim2
 * @param res: output matrix with dim2 x dim1
 * @param dim1: dimension 1 of input matrix
 * @param dim2: dimension 2 of input matrix
 */
__global__ void matrixTranspose_kenel(Complex_t *matrix, Complex_t *res, int dim1, int dim2);
/**
 * kernel function to swap the right and left half fo the input fftRes.
 * @param fftRes: input array with length 'ChirpSize'.
 */
__global__ void fftResSwap_kernel(Complex_t *fftRes, int size);

/**
 * Kernel function to perform fft for the input sequence in chunk.
 *
 * @param srcData: input complete sequence that need to be sliced into chunk
 * @param chunk_size: chunk size to perform
 * @param size: input complete size
 * @param stage: current stage of fft input
 * @param pow: chunk_size = 2 ^ pow
 */
__global__ void butterflyChunkFFT_kernel(Complex_t *srcData, int chunk_size, int size, int stage, int pow);

/**
 * Kernel function to perform swap for the fft res in chunk.
 * @param srcData: input complete data with length 'size'.
 * @param size: total size of input data
 * @param chunk_size: chunk size
 */
__global__ void fftResSwapChunk_kernel(Complex_t *srcData, int size, int chunk_size);

/**
 * kernel functiom to perform bit reverse swap for fft prepration
 * @param srcData: complete input sequence
 * @param size: length of input sequence
 * @param chunk_size: chunk size
 * @param pow: chunk_size = 2 ^ pow
 */
__global__ void bitReverseSwapChunk_kernel(Complex_t *srcData, int size, int chunk_size, int pow);

/**
 * Wrapper function to luanch cuda kernels
 * @param: Input: input_host: data read from .bin file in short format, with length 'size' = 'SampleSize * ChirpSize * numRx * 2'.
 * @param: Input: base_frame_rx0_device: allocated base frame rx0 data space in device side, with length 'SampleSize * ChirpSize'.
 * @param: Input: frame_reshaped_device: allocated reshaped frame data space in device side, with length 'size/2'.
 * @param: Input: size: int type indicates the total length of 'input_host'.
 * @param: Input: rx0_extended_size: int type indicates the length of 'rx0_extended_size'.
 */
void cudaAcceleration(double &speed, double &angle, double &distance,
                      double &speedTime, double &angleTime, double &distTime,
                      double &fftTime, double &preProcessingTime, double &findMaxTime, double &totalTime,
                      short *input_host, Complex_t *base_frame_device, Complex_t *frame_reshaped_device,
                      int size, int rx0_extended_size);

void launchPrePorc(short *input_host, Complex_t *base_frame_device, Complex_t *base_frame_rx0_device,
                   Complex_t *rx0_fft_input_device, Complex_t *frame_reshaped_device, int size,
                   int rx0_extended_size, cudaEvent_t *preProcEvt, cudaStream_t *preProcStream);

void launchDistProc(cudaEvent_t &preProcEvt, cudaEvent_t &distEvt, cudaStream_t &distStream,
                    Complex_t *distRes_fft_host_pinned, Complex_t *rx0_device, Complex_t *rx0_fft_input_device,
                    int rx0_extended_size);

void launchAngleProc(cudaEvent_t &preProcEvt, cudaEvent_t &distEvt, cudaEvent_t &angleEvt, cudaStream_t &angleStream,
                     Complex_t *frame_reshaped_device, Complex_t *base_frame_device,
                     Complex_t *distRes_fft_host_pinned, Complex_t *angleRes_host_pinned,
                     Complex_t *rx_fft_input_device, Complex_t *frame_reshaped_device_angle, Complex_t *base_frame_device_angle,
                     Complex_t *angle_weights_device, Complex_t *rx0_fft_input_device, Complex_t *angle_matrix_device, Complex_t *angle_matrix_res_device,
                     int rx0_extended_size, int maxIdx);

void launchSpeedProc(cudaEvent_t &preProcEvt, cudaEvent_t &speedEvt, cudaStream_t &speedStream,
                     Complex_t *frame_reshaped_device, Complex_t *base_frame_rx0_device, Complex_t *speedRes_host_pinned,
                     Complex_t *rx0_extended_fftRes_transpose, Complex_t *rx0_extended_fft_input_device,
                     int rx0_extended_size);

void cudaMultiStreamAcceleration(short *input_host, Complex_t *base_frame_device,
                                 Complex_t *frame_reshaped_device, Complex_t *rx0_fft_input_device_dist,
                                 Complex_t *rx_fft_input_device_angle, Complex_t *frame_reshaped_device_angle,
                                 Complex_t *base_frame_device_angle, Complex_t *angle_weights_device,
                                 Complex_t *rx0_fft_input_device_angle, Complex_t *angle_matrix_device, Complex_t *angle_matrix_res_device,
                                 Complex_t *rx0_extended_fftRes_transpose, Complex_t *rx0_extended_fft_input_device,
                                 int size, int rx0_extended_size);

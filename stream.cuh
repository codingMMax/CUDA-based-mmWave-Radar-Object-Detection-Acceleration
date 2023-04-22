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



void preProcessing_host(short *OriginalArray, Complex_t *Reshape, int size);
void printComplexCUDA(Complex_t *input, int start, int end, int size);
void printShortCUDA(short *input, int start, int end, int size);
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
 * Device function perform bit reverse and element swap for later fft
 * This function is doing the same as the kernel function, but is designed
 * to be called within a kernel before later fft.
*/
__device__ void bitReverseSwap_func(Complex_t *input, int size, int pow);

/**
 * kernel function perform butterfly computation fft for input data.
 * @param: input: 'data' complex input array with length 'size'.
 * @param: input: 'size' array length.
 * @param: input: 'stage' int number indicates the current stage for butterfly computation.
 * @param: input: 'pow' power of the input length = log2(size).
 */
__global__ void butterflyFFT_kernel(Complex_t *data, int size, int stage, int pow);

/**
 * device function perform butterfly computation fft for input data.
 * This function is the same as butterflyFFT_kernel function, but
 * it this is designed to be called within a kernel function for better
 * overall parallelsim.
*/
__device__ void butterflyFFT_func(Complex_t *data, int size, int stage, int pow);

/**
 * Kernel function that initialize the angle weights in 3 stages.
 * Stage1: initialize the 'angle_fft_buffer[i]' = 'reshaped_frame[i] - base_frame[i]'
 * Stage2: apply FFT for the angle_fft_buffer
 * Stage3: assign weight[1,2,3] = fft_res[maxAngleIdx] / extended_size
*/
__global__ void angleWeightInit_kernel(Complex_t *weights, Complex_t *rx0_fft_input_device, Complex_t *rx_fft_res, int maxAngleIdx, int size);

__global__ void angleMatrixInit_kernel(Complex_t *matrix,  int size);

__global__ void angleMatrixMul_kernel(Complex_t *angle_matrix, Complex_t *angle_weight, Complex_t *res, int num_angle_sample);

/**
 * Wrapper function to luanch cuda kernels
 * @param: Input: input_host: data read from .bin file in short format, with length 'size' = 'SampleSize * ChirpSize * numRx * 2'.
 * @param: Input: base_frame_rx0_device: allocated base frame rx0 data space in device side, with length 'SampleSize * ChirpSize'.
 * @param: Input: frame_buffer_device: allocated frame reshape buffer sapce in device side, with length 'size'.
 * @param: Input: frame_reshaped_device: allocated reshaped frame data space in device side, with length 'size/2'.
 * @param: Input: frame_reshaped_rx0_device: allocated reshaped frame rx0 data space in device side, with length 'rx0_extended_size'.
 * @param: Input: size: int type indicates the total length of 'input_host'.
 * @param: Input: rx0_extended_size: int type indicates the length of 'rx0_extended_size'.
 * @return: double format calculated distance of moving object
 *
 */
double cudaAcceleration(double& angleTime, double& distTime ,double &fftTime, double &preProcessingTime, double &findMaxTime, double &totalTime, short *input_host, Complex_t *base_frame_device, Complex_t *frame_buffer_device, Complex_t *frame_reshaped_device, Complex_t *frame_reshaped_rx0_device, int size, int rx0_extended_size);


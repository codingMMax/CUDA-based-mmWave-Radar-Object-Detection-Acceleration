#include "acceleration.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>

#define THREADS_PER_BLOCK 512
#define SampleSize 100   // the sample number in a chirp, suggesting it should be the power of 2
#define ChirpSize 128    // the chirp number in a frame, suggesting it should be the the power of 2
#define FrameSize 90     // the frame number
#define RxSize 4         // the rx size, which is usually 4
#define PI 3.14159265359 // pi

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = false)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",

                cudaGetErrorString(code), file, line);
        abort = true;
    }
    if (abort)
        exit(code);
}
#else
#define cudaCheckError(ans) ans
#endif

/**
 * get the extended size of input size
 */
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

__device__ static Complex_t cudaComplexSub(Complex_t a, Complex_t b)
{
    Complex_t temp;
    temp.real = a.real - b.real;
    temp.imag = a.imag - b.imag;
    return temp;
}

__device__ static Complex_t cudaComplexAdd(Complex_t a, Complex_t b)
{
    Complex_t temp;
    temp.real = a.real + b.real;
    temp.imag = a.imag + b.imag;
    return temp;
}

__device__ static Complex_t cudaComplexMul(Complex_t a, Complex_t b)
{
    Complex_t temp;
    temp.real = a.real * b.real - a.imag * b.imag;
    temp.imag = a.real * b.imag + a.imag * b.real;
    return temp;
}

__device__ static Complex_t makeComplex(double img, double real)
{
    Complex_t temp;
    temp.real = real;
    temp.imag = img;
    return temp;
}

__device__ static double cudaComplexMol(Complex_t *a)
{
    return sqrt(a->real * a->real + a->imag * a->imag);
}

/**
 * make the read short data type into complex number
 */
__global__ void cudaShort2Complex_kernel(short *input, Complex_t *buf, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int k = size / 4;
    short *srcPtr;
    Complex_t *destPtr;
    if (idx < k)
    {
        destPtr = buf + idx * 2;
        srcPtr = input + idx * 4;

        destPtr->real = (double)*(srcPtr);
        destPtr->imag = (double)*(srcPtr + 2);

        destPtr++;

        destPtr->real = (double)*(srcPtr + 1);
        destPtr->imag = (double)*(srcPtr + 3);
    }
}

/**
 * reshape the input complex buffer into specified format
 * input format: chirp0[rx0[sample0-100], rx1[sample0-100], rx2[sample0-100]..] chirp1[rx0[..]]
 * output format: rx0[chirp0[sample0-100],chirp1[0-100]...]
 */
__global__ void cudaComplexReshape_kernel(Complex_t *destArray, Complex_t *srcArray, int size)
{
    /**
     * chirpIdx: 0 - chirpSize ( < 128)
     * rxIdx: 0 - numRx ( < 4 )
     * sampleIdx: 0 - samepleSize( < 100)
     * srcIdx = chirpIdx * (RxSize * sampleSize) + rxIdx * sampleSize + sampleIdx
     * desidxx = rxIdx * (ChirpSize * SampleSize) + chirpIdx * SampleSize + sampleIdx
     */
    Complex_t *destPtr;
    Complex_t *srcPtr;
    int srcIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (srcIdx < size)
    {
        int chirpIdx = srcIdx / (RxSize * SampleSize);

        int rxIdx = (srcIdx - chirpIdx * RxSize * SampleSize) / SampleSize;

        int sampleIdx = srcIdx - chirpIdx * RxSize * SampleSize - rxIdx * SampleSize;

        int desidxx = rxIdx * ChirpSize * SampleSize + chirpIdx * SampleSize + sampleIdx;

        destPtr = destArray + desidxx;
        srcPtr = srcArray + srcIdx;

        destPtr->real = srcPtr->real;
        // destArray[desidxx].real = srcArray[srcIdx].real;
        // destPtr->real = (double)srcIdx;

        destPtr->imag = srcPtr->imag;
        // destArray[desidxx].imag = srcArray[srcIdx].imag;
    }
}

__global__ void cudaDataExtension_kernel(Complex_t *baseFrame, Complex_t *extendedBuffer, int oldSize, int extendedSize)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // set the extended part as zero
    if (idx < extendedSize && idx > oldSize)
    {
        extendedBuffer[idx].real = 0;
        extendedBuffer[idx].imag = 0;
    }
    if (idx < oldSize)
    {
        extendedBuffer[idx].imag = extendedBuffer[idx].imag - baseFrame[idx].imag;
        extendedBuffer[idx].real = extendedBuffer[idx].real - baseFrame[idx].real;
    }
}

__device__ int bitsReverse(int num, int bits)
{
    int rev = 0;
    for (int i = 0; i < bits; i++)
    {
        if (num & (1 << i))
        {
            rev |= 1 << ((bits - 1) - i);
        }
    }
    return rev;
}

__global__ void cudaBitsReverse_kernel(Complex_t *input, int size, int pow)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        // swap the position
        int pairIdx = bitsReverse(idx, pow);
        if (pairIdx > idx)
        {
            Complex_t temp = input[idx];
            input[idx] = input[pairIdx];
            input[pairIdx] = temp;
        }
    }
}

__global__ void cudaButterflyFFT_kernel(Complex_t *data, int size, int stage)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate pairIdx of the current element in the input data
    int pairIdx = idx << (stage + 1); // Equivalent to idx * pow(2, stage + 1)

    // Perform butterfly operation for each pair of elements at the current stage
    if (pairIdx < size)
    {
        // Load the two elements to be combined using butterfly operation
        Complex_t a = data[pairIdx];
        Complex_t b = data[pairIdx + (1 << stage)]; // Equivalent to pairIdx + pow(2, stage)

        // Calculate twiddle factor (complex exponential)
        float angle = -2.0f * PI * idx / (1 << (stage + 1)); // Equivalent to -2*pi*idx / pow(2, stage + 1)
        Complex_t twiddle;
        twiddle.real = cosf(angle);
        twiddle.imag = sinf(angle);

        // Perform butterfly operation
        Complex_t sum = cudaComplexAdd(a, b);
        Complex_t diff = cudaComplexMul(cudaComplexSub(a, b), twiddle);

        // Store the results back to global memory
        data[pairIdx] = sum;
        data[pairIdx + (1 << stage)] = diff;
    }
}

void printComplexCUDA(Complex_t *input, int start, int end, int size)
{
    printf("Displaying Complex Number processed from CUDA\n");

    Complex_t *cpuData = (Complex_t *)malloc(sizeof(Complex_t) * size);

    cudaMemcpy(cpuData, input, sizeof(Complex_t) * size, cudaMemcpyDeviceToHost);

    for (int i = start; i < end; i++)
    {
        printf("cudaComplex[%d] real: %.5f  img: %.5f\n", i, cpuData[i].real, cpuData[i].imag);
    }
    free(cpuData);
}

void printIntCUDA(int *input, int start, int end, int size)
{
    printf("Displaying short Number processed from CUDA\n");

    int *cpuData = (int *)malloc(sizeof(int) * size);

    cudaMemcpy(cpuData, input, sizeof(int) * size, cudaMemcpyDeviceToHost);
    for (int i = start; i < end; i++)
    {
        printf("intInput[%d] %d\n", i, cpuData[i]);
    }
    free(cpuData);
}

int GetBits(int n)
{
    int bits = 0;
    while (n >>= 1)
    {
        bits++;
    }
    return bits;
}

// /**
//  * CUDA Function that do the following things:
//  * 1. Reshape the input
//  * 2. Subtract the baseFrame
//  * 3. Apply FFT for Complext Element
//  * 4. Find the peak absolute amplitude in transfomed results
//  * 5. Return the peak value to host CPU for verification
//  */
double cudaProcessing(short *hostIn, Complex_t *host_baseFrame, int size)
{
    // accept the input data and reshape the data
    // define the kernel parameters
    printf("Entering CUDA processing functions\n");
    const int numBlocksShortKernel = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int complexSize = SampleSize * ChirpSize * RxSize;
    const int numBlocksComplexKernel = (complexSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // allocate cuda memory
    Complex_t *reshapedArray;
    Complex_t *buffer;
    short *deviceIn;
    printf("Allocating Memory buffer for reshaped Array with size of %lu byte\n", sizeof(Complex_t) * SampleSize * ChirpSize * RxSize);

    cudaCheckError(cudaMalloc((void **)&deviceIn, size * sizeof(short)));
    cudaCheckError(cudaMalloc((void **)&reshapedArray, sizeof(Complex_t) * SampleSize * ChirpSize * RxSize));
    cudaCheckError(cudaMalloc((void **)&buffer, sizeof(Complex_t) * SampleSize * ChirpSize * RxSize));
    cudaCheckError(cudaMemcpy(deviceIn, hostIn, size * sizeof(short), cudaMemcpyHostToDevice));

    /**
     * Memory copy check passed
     */

    printf("\nLaunching short2complex kernel\n");
    cudaShort2Complex_kernel<<<numBlocksShortKernel, THREADS_PER_BLOCK>>>(deviceIn, buffer, size);
    cudaDeviceSynchronize();
    // printComplexCUDA(buffer,25000,25010,SampleSize * ChirpSize * RxSize);

    /**
     * Above short2complex kernel is verified
     */
    printf("\nLaunching reshape kernel\n");
    cudaComplexReshape_kernel<<<numBlocksComplexKernel, THREADS_PER_BLOCK>>>(reshapedArray, buffer, complexSize);
    cudaDeviceSynchronize();

    // printf("\nreshaped Array\n");
    // printComplexCUDA(reshapedArray,37035,37039,SampleSize * ChirpSize * RxSize);
    // printComplexCUDA(reshapedArray,0,10,SampleSize * ChirpSize * RxSize);
    /**
     * Above reshape kernel is verified
     */
    int extendedSize = nextPow2(SampleSize * ChirpSize);
    const int numBlocksExtendedKernel = (extendedSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // only get the Rx0 data into extended buffer
    Complex_t *rx0ExtendedBuffer;
    Complex_t *baseFrame;

    cudaCheckError(cudaMalloc((void **)&rx0ExtendedBuffer, sizeof(Complex_t) * extendedSize));
    cudaCheckError(cudaMemcpy(rx0ExtendedBuffer, reshapedArray, SampleSize * ChirpSize * sizeof(Complex_t), cudaMemcpyDeviceToDevice));

    cudaCheckError(cudaMalloc((void **)&baseFrame, sizeof(Complex_t) * SampleSize * ChirpSize));
    cudaCheckError(cudaMemcpy(baseFrame, host_baseFrame, SampleSize * ChirpSize * sizeof(Complex_t), cudaMemcpyHostToDevice));
    // launch the data extension kernel
    cudaDataExtension_kernel<<<numBlocksExtendedKernel, THREADS_PER_BLOCK>>>(baseFrame, rx0ExtendedBuffer, SampleSize * ChirpSize, extendedSize);
    cudaDeviceSynchronize();
    // printComplexCUDA(rx0ExtendedBuffer,extendedSize - 20,extendedSize-10,extendedSize);

    /**
     * Above dataExtension kernel is verified
     */

    // launch the non-recursive FFT kernel
    int testSize = 16;
    Complex_t *fftRes;
    Complex_t *testInputHost = (Complex_t *)malloc(sizeof(Complex_t) * testSize);
    Complex_t *fftInputBuf;
    cudaCheckError(cudaMalloc((void **)&fftRes, sizeof(Complex_t) * testSize));
    cudaCheckError(cudaMalloc((void **)&fftInputBuf, sizeof(Complex_t) * testSize));
    for (int i = 0; i < testSize; i++)
    {
        testInputHost[i].real = i + 1;
        testInputHost[i].imag = 0;
    }
    cudaCheckError(cudaMemcpy(fftInputBuf, testInputHost, sizeof(Complex_t) * testSize, cudaMemcpyHostToDevice));
    int blocksFFTKernel = (testSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaBitsReverse_kernel<<<blocksFFTKernel, THREADS_PER_BLOCK>>>(fftInputBuf, testSize, log2(testSize));
    cudaDeviceSynchronize();
    for (int stage = 0; stage < log2(testSize); stage++)
    {
        cudaButterflyFFT_kernel<<<blocksFFTKernel, THREADS_PER_BLOCK>>>(fftInputBuf, testSize, stage);
        cudaDeviceSynchronize();
    }
    printf("FFT kernel Test\n");
    printComplexCUDA(fftInputBuf, 0, testSize, testSize);
    // printf("Ref kernel\n");
    // printComplexCUDA(fftResRef, 0, 10, extendedSize);

    cudaFree(deviceIn);
    cudaFree(buffer);
    cudaFree(fftRes);
    cudaFree(fftInputBuf);
    cudaFree(reshapedArray);
    cudaFree(rx0ExtendedBuffer);
    cudaFree(baseFrame);

    return 0.0;
}

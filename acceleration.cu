#include "acceleration.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>
#include <stdio.h>
#define THREADS_PER_BLOCK 512
#define SampleSize 100   // the sample number in a chirp, suggesting it should be the pow_tester of 2
#define ChirpSize 128    // the chirp number in a frame, suggesting it should be the the pow_tester of 2
#define FrameSize 90     // the frame number
#define RxSize 4         // the rx size, which is usually 4
#define PI 3.14159265359 // pi
#define fs 2.0e6         // sampling frequency
#define lightSpeed 3.0e08
#define mu 5.987e12 // FM slope

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = false)
{
    if (code != cudaSuccess)
    {
        // printf(stderr, "CUDA Error: %s at %s:%d\n",

        (cudaGetErrorString(code), file, line);
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

__device__ static double cudaComplexMol(Complex_t a)
{
    return sqrt(a.real * a.real + a.imag * a.imag);
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

__global__ void cudaBitsReverse_kernel(Complex_t *input, int size, int pow_test)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        // swap the position
        int pairIdx = bitsReverse(idx, pow_test);
        if (pairIdx > idx)
        {
            Complex_t temp = input[idx];
            input[idx] = input[pairIdx];
            input[pairIdx] = temp;
        }
    }
}

/**
 * @param: data: input complex_t sequence
 * @param: size: total size of input complex_t sequence
 * @param: stage: current stage of the FFT kernel, starting from 1
 */
__global__ void cudaButterflyFFT_kernel(Complex_t *data, int size, int stage, int pow_test)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Perform butterfly operation for each pair of elements at the current stage
    if (idx < size)
    {
        // calculate butterfly coefficient pow_tester
        int Wn_k = (1 << (pow_test - stage)) * idx % size;
        // butterfly coefficient = Wn ^ Wn_k
        // Wn = e^(-2j*pi/Size)
        // Wn ^ Wn_k = e ^ (-2j*pi*Wn_k/Size)
        double theta = -2 * PI * Wn_k / size;
        Complex_t twiddle = {cos(theta), sin(theta)};
        // calculate the pair index and multiplication factor
        int step = 1 << (stage - 1);
        int group_size = 1 << stage;
        int lower_bound = (idx / group_size) * group_size;
        int upper_bound = lower_bound + group_size;
        int pairIdx = idx + step;
        Complex_t product, sum;
        // product = p * a
        product = cudaComplexMul(twiddle, data[pairIdx]);
        // sum = q + (-1) * p * a
        sum = cudaComplexAdd(data[idx], product);
        // data[idx] = q - p*a
        if (pairIdx >= upper_bound)
        {
            pairIdx = idx - step;
            product = cudaComplexMul(twiddle, data[idx]);
            sum = cudaComplexAdd(data[pairIdx], product);
        }
        // write into the index position
        // __syncthreads();
        data[idx] = sum;

        //     // printf("idx %d twiddle.real %.3f twiddle.imag %.3f\n"
        //             "data.real %.3f data.imag %.3f \n"
        //             "product.real %.3f product.imag %.3f \n"
        //             "sum.real %.3f sum.imag %.3f\n\n", idx,
        //             twiddle.real, twiddle.imag, data[idx].real, data[idx].imag, product.real, product.imag, sum.real, sum.imag);
    }
    else
    {
        return;
    }
}
/**
 * kernel function to find the maxium value in the input FFT result and corresponding index
 * @param: data: input complex data
 * @param: size: input size
 * @param: maxValBuf: buffer to store the parallel selected local maxium values in each block, size = THREADS_PER_BLOCK
 * @param: maxIdxBuf: buffer to store the parallel selected maxium value corresponding index, size = THREADS_PER_BLOCK
 * @param: finalMaxValue: final sorted global maxium value
 * @param: finalMaxIdx: final sroted global maxium value corresponding index
 */
__global__ void cudaFindMax_kernel(Complex_t *data, int size, double *maxValBuf, int *maxIdxBuf, double *finalMaxValue, int *finalMaxIdx)
{
    int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (globalIdx < size)
    {
        // load the corresponding data into local block-shared memory
        double tmp = 0.0f;
        int maxIdx = 0;
        extern __shared__ Complex_t maxCache[THREADS_PER_BLOCK];
        extern __shared__ int idxCache[THREADS_PER_BLOCK];
        int localIdx = threadIdx.x;

        maxCache[localIdx] = data[globalIdx];
        idxCache[localIdx] = globalIdx;

        __syncthreads();
        if (localIdx == 0)
        {
            for (int i = 0; i < THREADS_PER_BLOCK; i++)
            {
                double absVal = cudaComplexMol(maxCache[i]);
                if (absVal > tmp)
                {
                    tmp = absVal;
                    maxIdx = idxCache[i];
                }
            }
            // write the compared value into global memory
            maxValBuf[blockIdx.x] = tmp;
            maxIdxBuf[blockIdx.x] = maxIdx;
        }
        __syncthreads();
        // once the maxVal array is filled. Find the max value in the maxVal array
        if (globalIdx < THREADS_PER_BLOCK)
        {
            extern __shared__ double finalMaxCache[THREADS_PER_BLOCK];
            extern __shared__ int finalIdxCache[THREADS_PER_BLOCK];
            finalMaxCache[globalIdx] = maxValBuf[globalIdx];
            finalIdxCache[globalIdx] = maxIdxBuf[globalIdx];
            __syncthreads();
            if (globalIdx == 0)
            {
                double finalMax = 0.0;
                int finalIdx;
                for (int i = 0; i < THREADS_PER_BLOCK; i++)
                {
                    if (finalMaxCache[i] > finalMax)
                    {
                        finalMax = finalMaxCache[i];
                        finalIdx = finalIdxCache[i];
                    }
                }
                *finalMaxValue = finalMax;
                *finalMaxIdx = finalIdx;
            }
        }
    }
}

void printComplexCUDA(Complex_t *input, int start, int end, int size)
{
    // printf("Displaying Complex Number processed from CUDA\n");

    Complex_t *cpuData = (Complex_t *)malloc(sizeof(Complex_t) * size);

    cudaMemcpy(cpuData, input, sizeof(Complex_t) * size, cudaMemcpyDeviceToHost);

    for (int i = start; i < end; i++)
    {
        // printf("cudaComplex[%d] real: %.5f  img: %.5f\n", i, cpuData[i].real, cpuData[i].imag);
    }
    free(cpuData);
}

void printIntCUDA(int *input, int start, int end, int size)
{
    // printf("Displaying short Number processed from CUDA\n");

    int *cpuData = (int *)malloc(sizeof(int) * size);

    cudaMemcpy(cpuData, input, sizeof(int) * size, cudaMemcpyDeviceToHost);
    for (int i = start; i < end; i++)
    {
        // printf("intInput[%d] %d\n", i, cpuData[i]);
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

void fftTest()
{
    int testSize = 16;
    Complex_t *testInputHost_test = (Complex_t *)malloc(sizeof(Complex_t) * testSize);
    Complex_t *fftInputBuf_test_device;
    cudaCheckError(cudaMalloc((void **)&fftInputBuf_test_device, sizeof(Complex_t) * testSize));
    for (int i = 0; i < testSize; i++)
    {
        testInputHost_test[i].real = i + 1;
        testInputHost_test[i].imag = 0;
    }
    int pow_test = 1, cnt_test = 0;
    while (pow_test < testSize)
    {
        pow_test <<= 1;
        cnt_test++;
    }

    cudaCheckError(cudaMemcpy(fftInputBuf_test_device, testInputHost_test, sizeof(Complex_t) * testSize, cudaMemcpyHostToDevice));
    int blocksFFTKernel_test = (testSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // printf("Pow %d log2(test size) %d\n", cnt_test, (int)log2((double)testSize));
    cudaBitsReverse_kernel<<<blocksFFTKernel_test, THREADS_PER_BLOCK>>>(fftInputBuf_test_device, testSize, log2(testSize));
    cudaDeviceSynchronize();
    printComplexCUDA(fftInputBuf_test_device, 0, testSize, testSize);

    for (int stage = 0; stage < log2(testSize); stage++)
    {
        // printf("stage %d\n", stage);
        cudaButterflyFFT_kernel<<<blocksFFTKernel_test, THREADS_PER_BLOCK>>>(fftInputBuf_test_device, testSize, stage + 1, cnt_test);
        cudaDeviceSynchronize();
    }
    // printf("FFT kernel Test\n");
    printComplexCUDA(fftInputBuf_test_device, 0, testSize, testSize);
    cudaFree(fftInputBuf_test_device);
    free(testInputHost_test);
}
int cudaFindAbsMax(Complex_t *ptr, int size)
{
    int maxidx = 0;
    double maxval = 0;
    double absval;
    for (int i = 0; i < size; i++)
    {
        Complex_t tmp = ptr[i];
        absval = sqrt(tmp.real * tmp.real + tmp.imag * tmp.imag);
        if (absval > maxval)
        {
            maxval = absval;
            maxidx = i;
        }
    }
    return maxidx;
}

// /**
//  * CUDA Function that do the following things:
//  * 1. Reshape the input
//  * 2. Subtract the baseFrame
//  * 3. Apply FFT for Complext Element
//  * 4. Find the peak absolute amplitude in transfomed results
//  * 5. Return the peak value to host CPU for verification
//  */
double cudaProcessing(short *input_host, Complex_t *host_baseFrame, int size)
{
    // accept the input data and reshape the data
    // define the kernel parameters
    // printf("Entering CUDA processing functions\n");

    Timer timer;

    double cudaBegin = timer.elapsed();
    const int numBlocksShortKernel = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int complexSize = SampleSize * ChirpSize * RxSize;
    const int numBlocksComplexKernel = (complexSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // allocate cuda memory
    Complex_t *reshapedArray;
    Complex_t *buffer;
    short *input_device;
    // printf("Allocating Memory buffer for reshaped Array with size of %lu byte\n", sizeof(Complex_t) * SampleSize * ChirpSize * RxSize);

    cudaCheckError(cudaMalloc((void **)&input_device, size * sizeof(short)));
    cudaCheckError(cudaMalloc((void **)&reshapedArray, sizeof(Complex_t) * SampleSize * ChirpSize * RxSize));
    cudaCheckError(cudaMalloc((void **)&buffer, sizeof(Complex_t) * SampleSize * ChirpSize * RxSize));
    cudaCheckError(cudaMemcpy(input_device, input_host, size * sizeof(short), cudaMemcpyHostToDevice));

    /**
     * Memory copy check passed
     */

    // printf("\nLaunching short2complex kernel\n");
    double cudaReshapeBegin = timer.elapsed();
    cudaShort2Complex_kernel<<<numBlocksShortKernel, THREADS_PER_BLOCK>>>(input_device, buffer, size);
    cudaDeviceSynchronize();
    // printComplexCUDA(buffer,25000,25010,SampleSize * ChirpSize * RxSize);

    /**
     * Above short2complex kernel is verified
     */
    // printf("\nLaunching reshape kernel\n");
    cudaComplexReshape_kernel<<<numBlocksComplexKernel, THREADS_PER_BLOCK>>>(reshapedArray, buffer, complexSize);
    cudaDeviceSynchronize();
    double cudaReshapeEnd = timer.elapsed();
    double cudaReshapeTime = cudaReshapeEnd - cudaReshapeBegin;
    // // printf("\nreshaped Array\n");
    // printComplexCUDA(reshapedArray,37035,37039,SampleSize * ChirpSize * RxSize);
    // printComplexCUDA(reshapedArray,0,10,SampleSize * ChirpSize * RxSize);
    /**
     * Above reshape kernel is verified
     */
    double cudaFrameExtensionBegin = timer.elapsed();
    int extendedSize = nextPow2(SampleSize * ChirpSize);
    const int numBlocksExtendedKernel = (extendedSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // only get the Rx0 data into extended buffer
    Complex_t *rx0ExtendedBuffer_device;
    Complex_t *baseFrame;

    cudaCheckError(cudaMalloc((void **)&rx0ExtendedBuffer_device, sizeof(Complex_t) * extendedSize));
    cudaCheckError(cudaMemcpy(rx0ExtendedBuffer_device, reshapedArray, SampleSize * ChirpSize * sizeof(Complex_t), cudaMemcpyDeviceToDevice));

    cudaCheckError(cudaMalloc((void **)&baseFrame, sizeof(Complex_t) * SampleSize * ChirpSize));
    cudaCheckError(cudaMemcpy(baseFrame, host_baseFrame, SampleSize * ChirpSize * sizeof(Complex_t), cudaMemcpyHostToDevice));
    // launch the data extension kernel
    cudaDataExtension_kernel<<<numBlocksExtendedKernel, THREADS_PER_BLOCK>>>(baseFrame, rx0ExtendedBuffer_device, SampleSize * ChirpSize, extendedSize);
    cudaDeviceSynchronize();
    // printComplexCUDA(rx0ExtendedBuffer_device,extendedSize - 20,extendedSize-10,extendedSize);
    double cudaFrameExtensionEnd = timer.elapsed();
    double cudaFrameExtensionTime = cudaFrameExtensionEnd - cudaFrameExtensionBegin;

    /**
     * Above dataExtension kernel is verified
     */

    // launch the non-recursive FFT kernel
    int cnt = 1, pow = 0;
    while (cnt < extendedSize)
    {
        cnt <<= 1;
        pow++;
    }
    // printf("\nLaunching FFT kernel \n");

    double cudaFFTbegin = timer.elapsed();

    Complex_t *fftInputBuf_device;
    cudaCheckError(cudaMalloc((void **)&fftInputBuf_device, sizeof(Complex_t) * extendedSize));
    cudaCheckError(cudaMemcpy(fftInputBuf_device, rx0ExtendedBuffer_device, sizeof(Complex_t) * extendedSize, cudaMemcpyDeviceToDevice));

    int blocksFFTKernel = (extendedSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaBitsReverse_kernel<<<blocksFFTKernel, THREADS_PER_BLOCK>>>(fftInputBuf_device, extendedSize, log2(extendedSize));
    cudaDeviceSynchronize();

    for (int stage = 0; stage < pow; stage++)
    {
        // // printf("stage %d\n", stage);
        cudaButterflyFFT_kernel<<<blocksFFTKernel, THREADS_PER_BLOCK>>>(fftInputBuf_device, extendedSize, stage + 1, pow);
    }
    // // printf("FFT result \n");
    // printComplexCUDA(fftInputBuf_device, extendedSize * 2 / 3, extendedSize * 2 / 3 + 10, extendedSize);

    /**
     * Above FFT kernel function is verified
     */
    // printf("\nLaunching findMax kernel\n");
    Complex_t *fftRes_host = (Complex_t *)malloc(sizeof(Complex_t) * extendedSize);
    cudaCheckError(cudaMemcpy(fftRes_host, fftInputBuf_device, sizeof(Complex_t) * extendedSize, cudaMemcpyDeviceToHost));
    double Fs_extend = fs * extendedSize / (ChirpSize * SampleSize);
    int maxDisIdx = cudaFindAbsMax(fftRes_host, floor(0.4 * extendedSize)) * (ChirpSize * SampleSize) / extendedSize;
    double maxDis = lightSpeed * (((double)maxDisIdx / extendedSize) * Fs_extend) / (2 * mu);
    // // printf("Finding maxDisIdx %d maxDis %.5f\n", maxDisIdx, maxDis);
    double cudaFFTend = timer.elapsed();
    double cudaFFTtime = cudaFFTend - cudaFFTbegin;

    double cudaEnd = timer.elapsed();
    double cudaTotalTime = cudaEnd - cudaBegin;

    printf("Inner CUDA Timing: total processing Time %.5f ms, FFT + findMax %.5f ms Reshape %.5f ms Extension %.5f ms\n", cudaTotalTime, cudaFFTtime, cudaReshapeTime, cudaFrameExtensionTime);

    free(fftRes_host);
    // double *maxValBuf_device;
    // int *maxIdxBuf_device;
    // double *finalMaxValue_device;
    // int *finalMaxIdx_device;
    // cudaCheckError(cudaMalloc((void **)&maxValBuf_device, sizeof(double) * THREADS_PER_BLOCK));
    // cudaCheckError(cudaMalloc((void **)&finalMaxValue_device, sizeof(double)));
    // cudaCheckError(cudaMalloc((void **)&maxIdxBuf_device, sizeof(int) * THREADS_PER_BLOCK));
    // cudaCheckError(cudaMalloc((void **)&finalMaxIdx_device, sizeof(int)));

    // cudaFindMax_kernel<<<blocksFFTKernel, THREADS_PER_BLOCK>>>(fftInputBuf_device, extendedSize, maxValBuf_device, maxIdxBuf_device, finalMaxValue_device, finalMaxIdx_device);

    // double *maxValue_host = (double *)malloc(sizeof(double));
    // int *maxIdx_host = (int *)malloc(sizeof(int));

    // cudaCheckError(cudaMemcpy(maxValue_host, finalMaxValue_device, sizeof(double), cudaMemcpyDeviceToHost));
    // cudaCheckError(cudaMemcpy(maxIdx_host, finalMaxIdx_device, sizeof(int), cudaMemcpyDeviceToHost));
    // int maxDisIdx = *maxIdx_host *
    // // printf("finding max value %.5f idx %d\n", *maxValue_host, *maxIdx_host);

    // cudaCheckError(cudaFree(maxValBuf_device));
    // cudaCheckError(cudaFree(maxIdxBuf_device));
    // cudaCheckError(cudaFree(finalMaxValue_device));
    // cudaCheckError(cudaFree(finalMaxIdx_device));

    cudaCheckError(cudaFree(input_device));
    cudaCheckError(cudaFree(buffer));
    cudaCheckError(cudaFree(fftInputBuf_device));
    cudaCheckError(cudaFree(reshapedArray));
    cudaCheckError(cudaFree(rx0ExtendedBuffer_device));
    cudaCheckError(cudaFree(baseFrame));

    return maxDis;
}

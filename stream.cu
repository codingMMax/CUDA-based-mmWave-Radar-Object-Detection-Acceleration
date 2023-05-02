#include "stream.cuh"

#define SHARED_MEM_SIZE sizeof(Complex_t) * ChirpSize

int multiStreamCnt = 0;

void preProcessing_host(short *OriginalArray, Complex_t *Reshape, int size)
{
    int i, j, k;
    Complex_t *buf_complex = (Complex_t *)malloc(size * sizeof(Complex_t) / 2);
    short *ptr;
    Complex_t *complex_ptr = buf_complex;
    // reshape into 2 form of complex
    for (i = 0; i < size; i += 4)
    {
        ptr = OriginalArray + i;
        complex_ptr->real = (double)*(ptr);
        complex_ptr->imag = (double)*(ptr + 2);
        complex_ptr++;
        complex_ptr->real = (double)*(ptr + 1);
        complex_ptr->imag = (double)*(ptr + 3);
        complex_ptr++;
    }
    Complex_t *Reshape_ptr;
    // change the sequence of the array, reshape it in form of Rx instead of frame
    for (i = 0; i < RxSize; i++)
    {
        for (j = 0; j < ChirpSize; j++)
        {
            for (k = 0; k < SampleSize; k++)
            {
                Reshape_ptr = (Reshape + i * ChirpSize * SampleSize + j * SampleSize + k);
                complex_ptr = (buf_complex + j * RxSize * SampleSize + i * SampleSize + k);
                Reshape_ptr->real = complex_ptr->real;
                Reshape_ptr->imag = complex_ptr->imag;
            }
        }
    }
    // printf("Processed Buffer\n");
    // for(int i = 25000; i < 25010; i ++){
    //     printf("buf_complex[%d] real: %.3f imag:%.3f \n",i,buf_complex[i].real, buf_complex[i].imag);
    // }

    free(buf_complex);
    return;
}

void printComplexCUDA(Complex_t *input, int start, int end, int size)
{
    // printf("Displaying Complex Number processed from CUDA\n");

    Complex_t *cpuData = (Complex_t *)malloc(sizeof(Complex_t) * size);

    cudaMemcpy(cpuData, input, sizeof(Complex_t) * size, cudaMemcpyDeviceToHost);

    for (int i = start; i < end; i++)
    {
        printf("cudaComplex[%d] real: %.9f  img: %.9f\n", i, cpuData[i].real, cpuData[i].imag);
    }
    free(cpuData);
}

void printShortCUDA(short *input, int start, int end, int size)
{
    // printf("Displaying short Number processed from CUDA\n");

    short *cpuData = (short *)malloc(sizeof(short) * size);

    cudaMemcpy(cpuData, input, sizeof(short) * size, cudaMemcpyDeviceToHost);
    for (int i = start; i < end; i++)
    {
        printf("shortInput[%d] %d\n", i, cpuData[i]);
    }
    free(cpuData);
}
/**
 * host function that can find the index of the maxium mol
 */
int findAbsMax(Complex_t *ptr, int size)
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
 * device function convert short data into paired complex number
 * @param input: short type input array with length 'size'.
 * @param buf: complex type output array with length 'size/2';
 * @param size: int type indicates the input arary length.
 */
__global__ void short2complex_kernel(short *input, Complex_t *buf, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int k = size / 4;
    if (idx < k)
    {
        short *srcPtr;
        Complex_t *destPtr;

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
 * kernel function reshape the input complex array into specified array layout
 * reshape the input complex buffer into specified format
 * input format: chirp0[rx0[sample0-100], rx1[sample0-100], rx2[sample0-100]..] chirp1[rx0[..]]
 * output format: rx0[chirp0[sample0-100],chirp1[0-100]...]
 * @param destArray: destination array hold the required array layout.
 * @param srcArray: src array with original array layout.
 * @param size: int variable indicates the array length.
 */
__global__ void complexReshape_kernel(Complex_t *destArray, Complex_t *srcArray, int size)
{
    Complex_t *destPtr;
    Complex_t *srcPtr;

    int srcIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (srcIdx < size)
    {
        int chirpIdx = srcIdx / (RxSize * SampleSize);

        int rxIdx = (srcIdx - chirpIdx * RxSize * SampleSize) / SampleSize;

        int sampleIdx = srcIdx - chirpIdx * RxSize * SampleSize - rxIdx * SampleSize;

        int destIdx = rxIdx * ChirpSize * SampleSize + chirpIdx * SampleSize + sampleIdx;

        destPtr = destArray + destIdx;
        srcPtr = srcArray + srcIdx;

        destPtr->real = srcPtr->real;
        // destArray[destIdx].real = srcArray[srcIdx].real;

        destPtr->imag = srcPtr->imag;
    }
}
/**
 * kernel function extend the rx0 data and fill in the fft input
 * @param baseFrame:  input base frame with length 'SampleSize * ChirpSize'
 * @param extendedBuffer: input buffer to store the extended data with length 'rx0_extended_size'.
 * @param reshaped_frame: already reshaped frame data, with length 'SampleSize * ChiprSize * numRx'.
 * @param rx0_extended_size: extended rx0 size = nextPow2(SampleSize * ChirpSize)
 *
 */
__global__ void rxExtension_kernel(Complex_t *baseFrame, Complex_t *extendedBuffer, Complex_t *reshaped_frame, int oldSize, int extendedSize)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // set the extended part as zero
    if (idx < extendedSize && idx >= oldSize)
    {
        extendedBuffer[idx].real = 0;
        extendedBuffer[idx].imag = 0;
    }
    if (idx < oldSize)
    {
        // printf("idx %d \n", idx);
        extendedBuffer[idx].imag = reshaped_frame[idx].imag - baseFrame[idx].imag;
        extendedBuffer[idx].real = reshaped_frame[idx].real - baseFrame[idx].real;
    }
}

/**
 * compute the reverse decimal number of input 'num' for given 'bits'
 */
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

/**
 * Kernel function perform bit reverse and element swap for later fft
 * @param input: 'input' complex input array, with length 'size'.
 * @param input: 'size' array length.
 * @param input: 'pow'  power of the input size = log2(size).
 */
__global__ void bitReverseSwap_kernel(Complex_t *input, int size, int pow)
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

/**
 * kernel function perform butterfly computation fft for input data.
 * @param input: 'data' complex input array with length 'size'.
 * @param input: 'size' array length.
 * @param input: 'stage' int number indicates the current stage for butterfly computation.
 * @param input: 'pow' power of the input length = log2(size).
 */
__global__ void butterflyFFT_kernel(Complex_t *data, int size, int stage, int pow)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Perform butterfly operation for each pair of elements at the current stage
    if (idx < size)
    {
        // calculate butterfly coefficient pow_tester
        int Wn_k = (1 << (pow - stage)) * idx % size;
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

        // data[idx] = q - p*a
        if (pairIdx >= upper_bound)
        {
            pairIdx = idx - step;
            product = cudaComplexMul(twiddle, data[idx]);
            sum = cudaComplexAdd(data[pairIdx], product);
        }
        else
        {
            product = cudaComplexMul(twiddle, data[pairIdx]);
            // sum = q + (-1) * p * a
            sum = cudaComplexAdd(data[idx], product);
        }
        // write into the index position
        // __syncthreads();
        data[idx] = sum;
    }
    else
    {
        return;
    }
}

/**
 * Kernel function to fill in the angle weights in device side
 */
__global__ void angleWeightInit_kernel(Complex_t *weights, Complex_t *rx0_fft_input_device, Complex_t *rx_fft_res, int maxAngleIdx, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < RxSize && idx >= 1)
    {
        weights[idx].real = rx_fft_res[(idx - 1) * size + maxAngleIdx].real / size;
        weights[idx].imag = rx_fft_res[(idx - 1) * size + maxAngleIdx].imag / size;
    }
    if (idx == 0)
    {
        weights[0].real = rx0_fft_input_device[maxAngleIdx].real / size;
        weights[0].imag = rx0_fft_input_device[maxAngleIdx].imag / size;
    }
}

/**
 * Kernel function to initialize the angle matrix
 * @param matrix: input matrix with dimension: RxSize * AngleSampleNum = 'size'.
 * row = RxSize
 * Col = AngleSampleNum
 */
__global__ void angleMatrixInit_kernel(Complex_t *matrix, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size)
    {
        int col = size / RxSize;
        int rowIdx = idx / col;
        int colIdx = idx - rowIdx * col;
        double phi = colIdx - 900;
        double LAMDA = 3.0e8 / 77e9;
        double D = 0.5 * LAMDA;
        double theta = -(double)rowIdx * 2 * PI * D * sin(phi / 1800.0 * PI) / LAMDA;
        matrix[idx].real = cos(theta);
        matrix[idx].imag = sin(theta);
    }
}

/**
 * angle matrix angle weights multiplication kernel
 * @param angle_matrix input matrix with dim: RxSize * num_angle_sample
 * @param angle_weight input matrix with dim: 1 * RxSize
 * @param res output matrix with dim: 1 * num_angle_sample
 */
__global__ void angleMatrixMul_kernel(Complex_t *angle_matrix, Complex_t *angle_weight, Complex_t *res, int num_angle_sample)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_angle_sample)
    {
        Complex_t tmp;
        tmp.real = 0;
        tmp.imag = 0;
        for (int i = 0; i < RxSize; i++)
        {
            tmp = cudaComplexAdd(tmp, cudaComplexMul(angle_weight[i], angle_matrix[i * num_angle_sample + idx]));
        }
        res[idx] = tmp;
    }
}

/**
 * kernel function for rx0 data padding
 * @param rx0_extended: extended rx0 data with length 'extended_sample_size * ChirpSize'.
 * @param rx0_non_extended: non-extended rx0 data with length 'ChirpSize * SampleSize'.
 * @param base_frame_rx0: rx0 data of base frame with length 'ChirpSize * SampleSize'.
 * @param extended_size: toal length of extended size = 'extended_sample_size * ChirpSize'.
 * @param non_extended_size: total length of non-extended size = 'SampleSize * ChirpSize'.
 * @param extended_sample_size: lenght of extende sample size = 'nexPow2(SampleSize)'.
 */
__global__ void rx0ChirpPadding_kernel(Complex_t *rx0_extended, Complex_t *rx0_non_extended, Complex_t *base_frame_rx0, int extended_size, int non_extended_size, int extended_sample_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < extended_size)
    {
        int chirpIdx = idx / extended_sample_size;
        int sampleIdx = idx - chirpIdx * extended_sample_size;
        if (sampleIdx < SampleSize)
        {
            // this sample is within the non-extended region
            Complex_t tmp;
            int non_extended_idx = chirpIdx * SampleSize + sampleIdx;
            tmp = cudaComplexSub(rx0_non_extended[non_extended_idx], base_frame_rx0[non_extended_idx]);
            rx0_extended[idx] = tmp;
        }
        else
        {
            rx0_extended[idx].real = 0;
            rx0_extended[idx].imag = 0;
        }
    }
}
/**
 * Kernel function to transpose the input matrix
 * @param matrix: input matrix with col x row
 * @param res: output matrix with row x col
 * @param col: dimension 1 of input matrix
 * @param row: dimension 2 of input matrix
 */
__global__ void matrixTranspose_kenel(Complex_t *matrix, Complex_t *res, int col, int row)
{
    int srcIdx = blockDim.x * blockIdx.x + threadIdx.x;
    // load the matrix value into shared block
    if (srcIdx < col * row)
    {
        // coordinates transform
        int colIdx = srcIdx / row;
        int rowIdx = srcIdx - colIdx * row;
        int destIdx = rowIdx * col + colIdx;

        // int rowIdx = srcIdx / col;
        // int colIdx = srcIdx - rowIdx * row;
        // int destIdx = rowIdx * col + colIdx;
        res[destIdx] = matrix[srcIdx];
    }
}

/**
 * kernel function to swap the right and left half fo the input fftRes.
 * @param fftRes: input array with length 'ChirpSize'.
 * @param size: input array length.
 */
__global__ void fftResSwap_kernel(Complex_t *fftRes)
{
    __shared__ Complex_t buffer[ChirpSize];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < ChirpSize)
    {
        buffer[idx] = fftRes[idx];
        __syncthreads();
        int mid = ChirpSize / 2;
        if (idx > mid)
        {
            fftRes[idx] = buffer[idx - mid];
        }
        else if (idx < mid)
        {
            fftRes[idx] = buffer[idx + mid];
        }
        else
        {
            fftRes[idx].real = 0;
            fftRes[idx].imag = 0;
        }
    }
}

/**
 * Kernel function to perform fft for the input sequence in chunk.
 *
 * @param srcData: input complete sequence that need to be sliced into chunk
 * @param chunk_size: chunk size to perform
 * @param size: input complete size
 * @param stage: current stage of fft input
 * @param pow: chunk_size = 2 ^ pow
 */
__global__ void butterflyChunkFFT_kernel(Complex_t *srcData, int chunk_size, int size, int stage, int pow)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int num_chunks = size / chunk_size;
    int chunkIdx = tid / chunk_size;
    if (chunkIdx < num_chunks)
    {
        // starting point of chunk data
        Complex_t *data = srcData + chunkIdx * chunk_size;
        // index for each thread within the specific chunk
        int idx = tid - chunkIdx * chunk_size;
        // calculate butterfly coefficient pow_tester
        /**
         * Same procedure as the butterfly fft kernel
         */
        int Wn_k = (1 << (pow - stage)) * idx % chunk_size;
        // butterfly coefficient = Wn ^ Wn_k
        // Wn = e^(-2j*pi/Size)
        // Wn ^ Wn_k = e ^ (-2j*pi*Wn_k/Size)
        double theta = -2 * PI * Wn_k / chunk_size;
        Complex_t twiddle = {cos(theta), sin(theta)};
        // calculate the pair index and multiplication factor
        int step = 1 << (stage - 1);
        int group_size = 1 << stage;
        int lower_bound = (idx / group_size) * group_size;
        int upper_bound = lower_bound + group_size;
        int pairIdx = idx + step;
        Complex_t product, sum;
        // product = p * a

        // data[idx] = q - p*a
        if (pairIdx >= upper_bound)
        {
            pairIdx = idx - step;
            product = cudaComplexMul(twiddle, data[idx]);
            sum = cudaComplexAdd(data[pairIdx], product);
        }
        else
        {
            product = cudaComplexMul(twiddle, data[pairIdx]);
            // sum = q + (-1) * p * a
            sum = cudaComplexAdd(data[idx], product);
        }
        // write into the index position
        // __syncthreads();
        data[idx] = sum;
    }
}

/**
 * Kernel function to perform swap for the fft res in chunk.
 * @param srcData: input complete data with length 'size'.
 * @param size: total size of input data
 * @param chunk_size: chunk size
 */
__global__ void fftResSwapChunk_kernel(Complex_t *srcData, int size, int chunk_size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int num_chunks = size / chunk_size;
    int chunkIdx = tid / chunk_size;
    if (chunkIdx < num_chunks)
    {
        __shared__ Complex_t buffer[ChirpSize];
        Complex_t *fftRes = srcData + chunkIdx * chunk_size;
        int idx = tid - chunk_size * chunkIdx;
        __syncthreads();
        int mid = ChirpSize / 2;
        if (idx > mid)
        {
            fftRes[idx] = buffer[idx - mid];
        }
        else if (idx < mid)
        {
            fftRes[idx] = buffer[idx + mid];
        }
        else
        {
            fftRes[idx].real = 0;
            fftRes[idx].imag = 0;
        }
    }
}
/**
 * kernel functiom to perform bit reverse swap for fft prepration
 * @param srcData: complete input sequence
 * @param size: length of input sequence
 * @param chunk_size: chunk size
 * @param pow: chunk_size = 2 ^ pow
 */
__global__ void bitReverseSwapChunk_kernel(Complex_t *srcData, int size, int chunk_size, int pow)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int num_chunks = size / chunk_size;
    int chunkIdx = tid / chunk_size;
    if (chunkIdx < num_chunks)
    {
        int idx = tid - chunkIdx * chunk_size;
        Complex_t *input = srcData + chunk_size * chunkIdx;
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

/**
 * Wrapper function to luanch cuda kernels
 * @param input_host: data read from .bin file in short format, with length 'size' = 'SampleSize * ChirpSize * numRx * 2'.
 * @param base_frame_rx0_device: allocated base frame rx0 data space in device side, with length 'SampleSize * ChirpSize'.
 * @param frame_buffer_device: allocated frame reshape buffer sapce in device side, with length 'size'.
 * @param frame_reshaped_device: allocated reshaped frame data space in device side, with length 'size/2'.
 * @param size: int type indicates the total length of 'input_host'.
 * @param rx0_extended_size: int type indicates the length of 'rx0_extended_size'.
 *
 */
void cudaAcceleration(double &speed, double &angle, double &distance, double &speedTime,
                      double &angleTime, double &distTime, double &fftTime, double &preProcessingTime,
                      double &findMaxTime, double &totalTime, short *input_host,
                      Complex_t *base_frame_device, Complex_t *frame_reshaped_device,
                      int size, int rx0_extended_size)
{

    cudaStream_t distStream;
    cudaStream_t angleStream;
    cudaStream_t speedStream;

    cudaCheckError(cudaStreamCreate(&distStream));
    cudaCheckError(cudaStreamCreate(&angleStream));
    cudaCheckError(cudaStreamCreate(&speedStream));

    /**
     * Pre-processing
     */
    Timer timer;
    double start = timer.elapsed();
    int num_blocks_preProcessing = (THREADS_PER_BLOCK + size - 1) / THREADS_PER_BLOCK;
    /**Allocate memory space for current frame*/
    short *input_device;
    cudaCheckError(cudaMalloc((void **)&input_device, sizeof(short) * size));
    cudaCheckError(cudaMemcpy(input_device, input_host, sizeof(short) * size, cudaMemcpyHostToDevice));

    Complex_t *rx0_fft_input_device;
    cudaCheckError(cudaMalloc((void **)&rx0_fft_input_device, sizeof(Complex_t) * rx0_extended_size));
    Complex_t *preProcessing_buffer;
    cudaCheckError(cudaMalloc((void **)&preProcessing_buffer, sizeof(Complex_t) * size / 2));

    short2complex_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(input_device, preProcessing_buffer, size);
    cudaDeviceSynchronize();

    num_blocks_preProcessing = (THREADS_PER_BLOCK + size / 2 - 1) / THREADS_PER_BLOCK;
    complexReshape_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(frame_reshaped_device, preProcessing_buffer, size / 2);
    cudaDeviceSynchronize();

    Complex_t *base_frame_rx0_device;
    cudaCheckError(cudaMalloc((void **)&base_frame_rx0_device, sizeof(Complex_t) * SampleSize * ChirpSize));
    cudaCheckError(cudaMemcpy(base_frame_rx0_device, base_frame_device, sizeof(Complex_t) * SampleSize * ChirpSize, cudaMemcpyDeviceToDevice));
    num_blocks_preProcessing = (THREADS_PER_BLOCK + rx0_extended_size - 1) / THREADS_PER_BLOCK;
    rxExtension_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(base_frame_rx0_device, rx0_fft_input_device, frame_reshaped_device, SampleSize * ChirpSize, rx0_extended_size);

    double preProcessingEnd = timer.elapsed();
    double fftStart = preProcessingEnd;
    preProcessingTime += preProcessingEnd - start;

    /**
     * Distance detection
     */

    int cnt = 1, pow = 0;
    while (cnt < rx0_extended_size)
    {
        cnt <<= 1;
        pow++;
    }
    cudaDeviceSynchronize();
    bitReverseSwap_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(rx0_fft_input_device, rx0_extended_size, pow);
    for (int i = 0; i < pow; i++)
    {
        butterflyFFT_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(rx0_fft_input_device, rx0_extended_size, i + 1, pow);
    }
    cudaDeviceSynchronize();

    double fftEnd = timer.elapsed();
    double findMaxStart = fftEnd;
    fftTime += fftEnd - fftStart;

    double Fs_extend = fs * rx0_extended_size / (ChirpSize * SampleSize);

    Complex_t *rx0_fft_res_host = (Complex_t *)malloc(sizeof(Complex_t) * rx0_extended_size);
    cudaCheckError(cudaMemcpy(rx0_fft_res_host, rx0_fft_input_device, sizeof(Complex_t) * rx0_extended_size, cudaMemcpyDeviceToHost));
    int maxIdx = findAbsMax(rx0_fft_res_host, floor(0.4 * rx0_extended_size));
    int maxDisIdx = maxIdx * (ChirpSize * SampleSize) / rx0_extended_size;

    double maxDis = lightSpeed * (((double)maxDisIdx / rx0_extended_size) * Fs_extend) / (2 * mu);
    distance = maxDis;

    double findMaxEnd = timer.elapsed();
    findMaxTime += findMaxEnd - findMaxStart;
    distTime += timer.elapsed() - start;

    /**
     * Angle Detection
     * Stage1: Extend the rx1 rx2 and rx3 data as rx0
     * Stage2: FFT corresponding rx data
     * Stage3: find max assign the value
     * Stage4: MMM for angle weights
     */
    // initialize the angle weights parameters
    double angleBegin = timer.elapsed();
    int maxAngleIdx = maxIdx;

    // stage1 allocate memory for rx1 rx2 and rx3
    Complex_t *rx_fft_input_device;
    cudaCheckError(cudaMalloc((void **)&rx_fft_input_device, sizeof(Complex_t) * rx0_extended_size * (RxSize - 1)));
    int num_blocks_angle = (rx0_extended_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int i = 0; i < RxSize - 1; i++)
    {
        double anglePreProc = timer.elapsed();
        Complex_t *rx_fft_input_device_ptr = rx_fft_input_device + i * rx0_extended_size;
        Complex_t *reshaped_frame_devce_ptr = frame_reshaped_device + (i + 1) * ChirpSize * SampleSize;
        Complex_t *base_frame_device_ptr = base_frame_device + (i + 1) * ChirpSize * SampleSize;
        // copy the reshaped frame data into corresponding rx holder in device
        rxExtension_kernel<<<num_blocks_angle, THREADS_PER_BLOCK>>>(base_frame_device_ptr, rx_fft_input_device_ptr, reshaped_frame_devce_ptr, SampleSize * ChirpSize, rx0_extended_size);
        preProcessingTime += (timer.elapsed() - anglePreProc);

        // above extension operation is verified
        // apply butterfly FFT
        double angleFFTBegin = timer.elapsed();
        cudaDeviceSynchronize();
        bitReverseSwap_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(rx_fft_input_device_ptr, rx0_extended_size, pow);
        // cudaDeviceSynchronize();
        for (int stage = 0; stage < pow; stage++)
        {
            butterflyFFT_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(rx_fft_input_device_ptr, rx0_extended_size, stage + 1, pow);
        }

        fftTime += (timer.elapsed() - angleFFTBegin);
    }
    cudaDeviceSynchronize();

    // above operations are verified
    // assign values to angle weigths
    Complex_t *angle_weights_device;
    cudaCheckError(cudaMalloc((void **)&angle_weights_device, sizeof(Complex_t) * RxSize));
    angleWeightInit_kernel<<<1, RxSize>>>(angle_weights_device, rx0_fft_input_device, rx_fft_input_device, maxAngleIdx, rx0_extended_size);

    // Stage4 MMM: Angle Matrix X angle_weights
    Complex_t *angle_matrix_device;
    int angle_sample_num = 180 / 0.1 + 1;
    cudaCheckError(cudaMalloc((void **)&angle_matrix_device, RxSize * angle_sample_num * sizeof(Complex_t)));
    num_blocks_angle = (RxSize * angle_sample_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaDeviceSynchronize();

    angleMatrixInit_kernel<<<num_blocks_angle, THREADS_PER_BLOCK>>>(angle_matrix_device, RxSize * angle_sample_num);
    cudaDeviceSynchronize();

    num_blocks_angle = (angle_sample_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    Complex_t *angle_matrix_res_device;
    Complex_t *angle_matrix_res_host;
    cudaCheckError(cudaMalloc((void **)&angle_matrix_res_device, sizeof(Complex_t) * angle_sample_num));
    angle_matrix_res_host = (Complex_t *)malloc(sizeof(Complex_t) * angle_sample_num);
    angleMatrixMul_kernel<<<num_blocks_angle, THREADS_PER_BLOCK>>>(angle_matrix_device, angle_weights_device, angle_matrix_res_device, angle_sample_num);
    cudaDeviceSynchronize();

    double angleFindMax = timer.elapsed();
    cudaCheckError(cudaMemcpy(angle_matrix_res_host, angle_matrix_res_device, sizeof(Complex_t) * angle_sample_num, cudaMemcpyDeviceToHost));
    maxAngleIdx = findAbsMax(angle_matrix_res_host, angle_sample_num);
    findMaxTime += (timer.elapsed() - angleFindMax);
    // above operations are verified
    double maxAgl = (maxAngleIdx - 900.0) / 10.0;
    angle = maxAgl;
    // printf("maxAgl %.9f\n",maxAgl);
    angleTime += (timer.elapsed() - angleBegin);

    /**
     * Speed Detection
     * 1. stage1 padding each chirp data in rx0 to be 2^n
     * 2. stage2 apply fft for each padded chirp of rx0 in chirp dimension
     * 3. stage3 transpose the transformed fft results
     * 4. stage4 apply fft in sample dimension and swap the front half and back half
     */

    // allocate memory sapce for the padding rx0
    double speedBegin = timer.elapsed();

    Complex_t *rx0_extended_fft_input_device;
    cudaCheckError(cudaMalloc((void **)&rx0_extended_fft_input_device, sizeof(Complex_t) * rx0_extended_size));
    int extended_sample_size = nextPow2(SampleSize);
    int num_blocks_speed = (rx0_extended_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // stage1 using the frame_reshaped_device used as the ptr to original non-extended rx0 frame data
    double speedPreProc = timer.elapsed();
    rx0ChirpPadding_kernel<<<num_blocks_speed, THREADS_PER_BLOCK>>>(rx0_extended_fft_input_device, frame_reshaped_device, base_frame_rx0_device, rx0_extended_size, SampleSize * ChirpSize, extended_sample_size);
    cudaDeviceSynchronize();
    preProcessingTime += (timer.elapsed() - speedPreProc);
    // above extension is verified

    cnt = 1;
    pow = 0;
    while (cnt < extended_sample_size)
    {
        cnt <<= 1;
        pow++;
    }
    // stage2 apply fft for each padded chirp of rx0
    double speedFFTBegin = timer.elapsed();

    num_blocks_speed = (extended_sample_size * ChirpSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    bitReverseSwapChunk_kernel<<<num_blocks_speed, THREADS_PER_BLOCK>>>(rx0_extended_fft_input_device, extended_sample_size * ChirpSize, extended_sample_size, pow);
    cudaDeviceSynchronize();

    for (int stage = 0; stage < pow; stage++)
    {
        butterflyChunkFFT_kernel<<<num_blocks_speed, THREADS_PER_BLOCK>>>(rx0_extended_fft_input_device, extended_sample_size, extended_sample_size * ChirpSize, stage + 1, pow);
    }
    cudaDeviceSynchronize();

    // above chunk vitreverseSwap is verified

    double speedMarker = timer.elapsed() - speedFFTBegin;

    fftTime += (timer.elapsed() - speedFFTBegin);

    // above FFT is verified
    // stage3 transpose the fft res
    Complex_t *rx0_extended_fftRes_transpose;
    cudaCheckError(cudaMalloc((void **)&rx0_extended_fftRes_transpose, sizeof(Complex_t) * rx0_extended_size));
    num_blocks_speed = (extended_sample_size * ChirpSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    matrixTranspose_kenel<<<num_blocks_speed, THREADS_PER_BLOCK>>>(rx0_extended_fft_input_device, rx0_extended_fftRes_transpose, SampleSize, ChirpSize);
    cudaDeviceSynchronize();

    // stage4 apply fft for the transposed data
    // and swap the right and left half
    speedFFTBegin = timer.elapsed();

    // above operations is verified

    pow = 0;
    cnt = 1;
    while (cnt < ChirpSize)
    {
        cnt <<= 1;
        pow++;
    }
    num_blocks_speed = (extended_sample_size * ChirpSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    bitReverseSwapChunk_kernel<<<num_blocks_speed, THREADS_PER_BLOCK>>>(rx0_extended_fftRes_transpose, extended_sample_size * ChirpSize, ChirpSize, pow);

    for (int stage = 0; stage < pow; stage++)
    {
        butterflyChunkFFT_kernel<<<num_blocks_speed, THREADS_PER_BLOCK>>>(rx0_extended_fftRes_transpose, ChirpSize, rx0_extended_size, stage + 1, pow);
    }

    cudaDeviceSynchronize();
    fftResSwapChunk_kernel<<<num_blocks_speed, THREADS_PER_BLOCK>>>(rx0_extended_fftRes_transpose, rx0_extended_size, ChirpSize);
    cudaDeviceSynchronize();

    speedMarker = timer.elapsed() - speedFFTBegin;

    fftTime += (timer.elapsed() - speedFFTBegin);

    Complex_t *speed_fft_res_host = (Complex_t *)malloc(sizeof(Complex_t) * rx0_extended_size);

    cudaCheckError(cudaMemcpy(speed_fft_res_host, rx0_extended_fftRes_transpose, sizeof(Complex_t) * rx0_extended_size, cudaMemcpyDeviceToHost));

    double speedFindMax = timer.elapsed();

    int maxSpeedIdx = findAbsMax(speed_fft_res_host, ChirpSize * SampleSize) % ChirpSize;

    findMaxTime += (timer.elapsed() - speedFindMax);

    double fr = 1e6 / 64;
    double LAMDA = 3.0e08 / 77e9;
    double maxSpeed = ((maxSpeedIdx)*fr / ChirpSize - fr / 2) * LAMDA / 2;
    speed = maxSpeed;
    speedTime += (timer.elapsed() - speedBegin);
    // printf("maxSpeed %.3f m/s\n", maxSpeed);

    totalTime += (timer.elapsed() - start);

    cudaCheckError(cudaStreamDestroy(distStream));
    cudaCheckError(cudaStreamDestroy(angleStream));
    cudaCheckError(cudaStreamDestroy(speedStream));

    free(rx0_fft_res_host);
    free(angle_matrix_res_host);
    free(speed_fft_res_host);

    cudaCheckError(cudaFree(rx_fft_input_device));
    cudaCheckError(cudaFree(base_frame_rx0_device));
    cudaCheckError(cudaFree(angle_weights_device));
    cudaCheckError(cudaFree(angle_matrix_device));
    cudaCheckError(cudaFree(angle_matrix_res_device));
    cudaCheckError(cudaFree(input_device));

    cudaCheckError(cudaFree(rx0_extended_fft_input_device));
    cudaCheckError(cudaFree(rx0_fft_input_device));

    cudaCheckError(cudaFree(rx0_extended_fftRes_transpose));
    cudaCheckError(cudaFree(preProcessing_buffer));
}

void launchPrePorc(short *input_host, Complex_t *base_frame_device, Complex_t *base_frame_rx0_device,
                   Complex_t *rx0_fft_input_device, Complex_t *frame_reshaped_device, int size,
                   int rx0_extended_size, cudaEvent_t &preProcEvt, cudaStream_t &preProcStream)
{
    // printf("launch preprocessing\n");

    int num_blocks = (THREADS_PER_BLOCK + size - 1) / THREADS_PER_BLOCK;
    short *input_device;
    // pre processing stream
    // cudaCheckError(cudaStreamCreat(&preProcEvt));

    cudaCheckError(cudaMalloc((void **)&input_device, sizeof(short) * size));
    cudaCheckError(cudaMemcpy(input_device, input_host, sizeof(short) * size, cudaMemcpyHostToDevice));

    Complex_t *preProcessing_buffer;
    cudaCheckError(cudaMalloc((void **)&preProcessing_buffer, sizeof(Complex_t) * size / 2));
    long memSize = sizeof(Complex_t) * size / 2 + sizeof(short) * size;
    // printf("memSize allocated in preProcessing %.3f KiBytes\n", (double)(memSize / 1024));

    short2complex_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, preProcStream>>>(input_device, preProcessing_buffer, size);

    num_blocks = (THREADS_PER_BLOCK + size / 2 - 1) / THREADS_PER_BLOCK;

    cudaDeviceSynchronize();
    complexReshape_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, preProcStream>>>(frame_reshaped_device, preProcessing_buffer, size / 2);

    // Complex_t *base_frame_rx0_device;
    // cudaCheckError(cudaMalloc((void **)&base_frame_rx0_device, sizeof(Complex_t) * SampleSize * ChirpSize));
    cudaCheckError(cudaMemcpy(base_frame_rx0_device, base_frame_device, sizeof(Complex_t) * SampleSize * ChirpSize, cudaMemcpyDeviceToDevice));
    num_blocks = (THREADS_PER_BLOCK + rx0_extended_size - 1) / THREADS_PER_BLOCK;
    cudaDeviceSynchronize();
    rxExtension_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, preProcStream>>>(base_frame_rx0_device, rx0_fft_input_device, frame_reshaped_device, SampleSize * ChirpSize, rx0_extended_size);

    cudaCheckError(cudaFree(input_device));
    cudaCheckError(cudaFree(preProcessing_buffer));
    cudaCheckError(cudaEventRecord(preProcEvt, preProcStream));
}

void launchDistProc(cudaEvent_t &preProcEvt, cudaEvent_t &distEvt, cudaStream_t &distStream,
                    Complex_t *distRes_fft_host_pinned, Complex_t *rx0_device, Complex_t *rx0_fft_input_device,
                    int rx0_extended_size)
{
    // printf("launch distance processing\n");

    // block distStream untill the preprocessing event is happend
    cudaCheckError(cudaStreamWaitEvent(distStream, preProcEvt));
    double memSize = 0;
    /**
     * copy memory buffers of DISTANCE PROC asyncrhoniously
     */
    // Complex_t *rx0_fft_input_device;
    // cudaCheckError(cudaMallocAsync((void **)&rx0_fft_input_device, sizeof(Complex_t) * rx0_extended_size, distStream));
    // cudaCheckError(cudaMalloc((void **)&rx0_fft_input_device, sizeof(Complex_t) * rx0_extended_size));

    cudaCheckError(cudaMemcpyAsync(rx0_fft_input_device, rx0_device, sizeof(Complex_t) * rx0_extended_size, cudaMemcpyDeviceToDevice, distStream));
    // cudaCheckError(cudaMemcpy(rx0_fft_input_device, rx0_device, sizeof(Complex_t) * rx0_extended_size, cudaMemcpyDeviceToDevice));
    memSize += 2 * sizeof(Complex_t) * rx0_extended_size;

    // distance calculation starts
    int cnt = 1;
    int pow = 0;
    while (cnt < rx0_extended_size)
    {
        cnt <<= 1;
        pow++;
    }
    int num_blocks = (THREADS_PER_BLOCK + rx0_extended_size - 1) / THREADS_PER_BLOCK;
    bitReverseSwap_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, distStream>>>(rx0_fft_input_device, rx0_extended_size, pow);
    // bitReverseSwap_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(rx0_fft_input_device, rx0_extended_size, pow);
    // cudaCheckError(cudaDeviceSynchronize());
    for (int i = 0; i < pow; i++)
    {
        butterflyFFT_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, distStream>>>(rx0_fft_input_device, rx0_extended_size, i + 1, pow);
        // butterflyFFT_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(rx0_fft_input_device, rx0_extended_size, i + 1, pow);
    }
    // cudaCheckError(cudaDeviceSynchronize());

    /**
     * DIST RES write back
     * copy the device result to host pinned memory
     */
    cudaCheckError(cudaMemcpyAsync(distRes_fft_host_pinned, rx0_fft_input_device, sizeof(Complex_t) * rx0_extended_size, cudaMemcpyDeviceToHost, distStream));
    // cudaCheckError(cudaFreeAsync(rx0_fft_input_device, distStream));
    // cudaCheckError(cudaMemcpy(distRes_fft_host_pinned, rx0_fft_input_device, sizeof(Complex_t) * rx0_extended_size, cudaMemcpyDeviceToHost));
    // cudaCheckError(cudaFree(rx0_fft_input_device));
    // printf("dist memSize %.3f KiBytes\n", (double)(memSize / 1024));
    cudaCheckError(cudaEventRecord(distEvt));
}

void launchAngleProc(cudaEvent_t &preProcEvt, cudaEvent_t &distEvt, cudaEvent_t &angleEvt, cudaStream_t &angleStream,
                     Complex_t *frame_reshaped_device, Complex_t *base_frame_device,
                     Complex_t *distRes_fft_host_pinned, Complex_t *angleRes_host_pinned,
                     Complex_t *rx_fft_input_device, Complex_t *frame_reshaped_device_angle, Complex_t *base_frame_device_angle,
                     Complex_t *angle_weights_device, Complex_t *rx0_fft_input_device, Complex_t *angle_matrix_device, Complex_t *angle_matrix_res_device,
                     int rx0_extended_size, int maxIdx)
{
    // printf("launch angle processing\n");

    long memSize = 0;
    // block angle stream untill the pre processing event is occured
    cudaCheckError(cudaStreamWaitEvent(angleStream, preProcEvt));
    // block angle stream untill the dsitance event is occured
    cudaCheckError(cudaStreamWaitEvent(angleStream, distEvt));

    /**
     *  copy memory buffers of ANGLE PROCE asyncrhoniously
     */
    // Complex_t *rx_fft_input_device;
    // Complex_t *frame_reshaped_device_angle;
    // Complex_t *base_frame_device_angle;
    // cudaCheckError(cudaMallocAsync((void **)&rx_fft_input_device, sizeof(Complex_t) * rx0_extended_size * (RxSize - 1), angleStream));
    // cudaCheckError(cudaMallocAsync((void **)&base_frame_device_angle, sizeof(Complex_t) * SampleSize * ChirpSize * RxSize, angleStream));
    // cudaCheckError(cudaMallocAsync((void **)&frame_reshaped_device_angle, sizeof(Complex_t) * SampleSize * ChirpSize * RxSize, angleStream));
    memSize += (sizeof(Complex_t) * rx0_extended_size * (RxSize - 1) + sizeof(Complex_t) * SampleSize * ChirpSize * RxSize + sizeof(Complex_t) * SampleSize * ChirpSize * RxSize);

    cudaCheckError(cudaMemcpyAsync(frame_reshaped_device_angle, frame_reshaped_device, sizeof(Complex_t) * SampleSize * ChirpSize * RxSize, cudaMemcpyDeviceToDevice, angleStream));
    cudaCheckError(cudaMemcpyAsync(base_frame_device_angle, base_frame_device, sizeof(Complex_t) * SampleSize * ChirpSize * RxSize, cudaMemcpyDeviceToDevice, angleStream));

    int num_blocks = (rx0_extended_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int cnt = 1, pow = 0;
    while (cnt < rx0_extended_size)
    {
        cnt <<= 1;
        pow++;
    }

    for (int i = 0; i < RxSize - 1; i++)
    {
        Complex_t *rx_fft_input_device_ptr = rx_fft_input_device + i * rx0_extended_size;
        Complex_t *reshaped_frame_device_ptr = frame_reshaped_device_angle + (i + 1) * ChirpSize * SampleSize;
        Complex_t *base_frame_device_ptr = base_frame_device + (i + 1) * ChirpSize * SampleSize;

        rxExtension_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, angleStream>>>(base_frame_device_ptr, rx_fft_input_device_ptr, reshaped_frame_device_ptr, SampleSize * ChirpSize, rx0_extended_size);
        bitReverseSwap_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, angleStream>>>(rx_fft_input_device_ptr, rx0_extended_size, pow);
        for (int stage = 0; stage < pow; stage++)
        {
            butterflyFFT_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, angleStream>>>(rx_fft_input_device_ptr, rx0_extended_size, stage + 1, pow);
        }
    }

    // Complex_t *angle_weights_device;
    // Complex_t *rx0_fft_input_device;
    // Complex_t *angle_matrix_device;
    // Complex_t *angle_matrix_res_device;
    int angle_sample_num = 180 / 0.1 + 1;
    // cudaCheckError(cudaMallocAsync((void **)&angle_weights_device, sizeof(Complex_t) * RxSize, angleStream));
    // cudaCheckError(cudaMallocAsync((void **)&rx0_fft_input_device, sizeof(Complex_t) * rx0_extended_size, angleStream));
    // cudaCheckError(cudaMallocAsync((void **)&angle_matrix_device, sizeof(Complex_t) * RxSize * angle_sample_num, angleStream));
    // cudaCheckError(cudaMallocAsync((void **)&angle_matrix_res_device, sizeof(Complex_t) * angle_sample_num, angleStream));

    memSize += (sizeof(Complex_t) * RxSize + sizeof(Complex_t) * RxSize + sizeof(Complex_t) * rx0_extended_size + sizeof(Complex_t) * angle_sample_num);

    cudaCheckError(cudaMemcpyAsync(rx0_fft_input_device, distRes_fft_host_pinned, sizeof(Complex_t) * rx0_extended_size, cudaMemcpyDeviceToDevice, angleStream));

    angleWeightInit_kernel<<<1, RxSize, 0, angleStream>>>(angle_weights_device, rx0_fft_input_device, rx_fft_input_device, maxIdx, rx0_extended_size);
    num_blocks = (angle_sample_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    angleMatrixInit_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, angleStream>>>(angle_matrix_device, RxSize * angle_sample_num);
    angleMatrixMul_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, angleStream>>>(angle_matrix_device, angle_weights_device, angle_matrix_res_device, angle_sample_num);
    /**
     * ANGLE RES write back
     * copy the device result to host pinned memory
     */
    // printf("angle memSize %.3f KiBytes\n", (double)(memSize / 1024));
    cudaCheckError(cudaMemcpyAsync(angleRes_host_pinned, angle_matrix_res_device, sizeof(Complex_t) * angle_sample_num, cudaMemcpyDeviceToHost, angleStream));
    // cudaCheckError(cudaFreeAsync(rx_fft_input_device, angleStream));
    // cudaCheckError(cudaFreeAsync(frame_reshaped_device_angle, angleStream));
    // cudaCheckError(cudaFreeAsync(base_frame_device_angle, angleStream));
    // cudaCheckError(cudaFreeAsync(angle_weights_device, angleStream));
    // cudaCheckError(cudaFreeAsync(rx0_fft_input_device, angleStream));
    // cudaCheckError(cudaFreeAsync(angle_matrix_device, angleStream));
    // cudaCheckError(cudaFreeAsync(angle_matrix_res_device, angleStream));

    cudaCheckError(cudaEventRecord(angleEvt, angleStream));
}

void launchSpeedProc(cudaEvent_t &preProcEvt, cudaEvent_t &speedEvt, cudaStream_t &speedStream,
                     Complex_t *frame_reshaped_device, Complex_t *base_frame_rx0_device, Complex_t *speedRes_host_pinned,
                     Complex_t *rx0_extended_fftRes_transpose, Complex_t *rx0_extended_fft_input_device,
                     int rx0_extended_size)
{
    // printf("launch speed processing\n");

    long memSize = 0;
    cudaCheckError(cudaStreamWaitEvent(speedStream, preProcEvt));

    /**
     *  copy memory buffers of SPEED PROC asyncrhoniously
     */
    // Complex_t *rx0_extended_fft_input_device;

    // cudaCheckError(cudaMallocAsync((void **)&rx0_extended_fft_input_device, sizeof(Complex_t) * extended_sample_size * ChirpSize, speedStream));

    int extended_sample_size = nextPow2(SampleSize);

    memSize += sizeof(Complex_t) * extended_sample_size * ChirpSize;

    int num_blocks = (rx0_extended_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    rx0ChirpPadding_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, speedStream>>>(rx0_extended_fft_input_device, frame_reshaped_device, base_frame_rx0_device, rx0_extended_size, SampleSize * ChirpSize, extended_sample_size);
    int cnt = 1;
    int pow = 0;
    while (cnt < extended_sample_size)
    {
        cnt <<= 1;
        pow++;
    }
    num_blocks = (extended_sample_size * ChirpSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    bitReverseSwapChunk_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, speedStream>>>(rx0_extended_fft_input_device, extended_sample_size * ChirpSize, extended_sample_size, pow);
    for (int stage = 0; stage < pow; stage++)
    {
        butterflyChunkFFT_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, speedStream>>>(rx0_extended_fft_input_device, extended_sample_size, extended_sample_size * ChirpSize, stage + 1, pow);
    }

    // Complex_t *rx0_extended_fftRes_transpose;
    // cudaCheckError(cudaMallocAsync((void **)&rx0_extended_fftRes_transpose, sizeof(Complex_t) * extended_sample_size * ChirpSize, speedStream));

    memSize += sizeof(Complex_t) * extended_sample_size * ChirpSize;

    matrixTranspose_kenel<<<num_blocks, THREADS_PER_BLOCK, 0, speedStream>>>(rx0_extended_fft_input_device, rx0_extended_fftRes_transpose, SampleSize, ChirpSize);

    cnt = 1;
    pow = 0;
    while (cnt < ChirpSize)
    {
        cnt <<= 1;
        pow++;
    }

    bitReverseSwapChunk_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, speedStream>>>(rx0_extended_fftRes_transpose, rx0_extended_size * ChirpSize, ChirpSize, pow);
    for (int stage = 0; stage < pow; stage++)
    {
        butterflyChunkFFT_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, speedStream>>>(rx0_extended_fftRes_transpose, ChirpSize, ChirpSize * extended_sample_size, stage + 1, pow);
    }

    fftResSwapChunk_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, speedStream>>>(rx0_extended_fftRes_transpose, ChirpSize * extended_sample_size, ChirpSize);

    // printf("Speed Mem allocating %.3f KiBytes\n", (double)(memSize / 1024));

    /**
     * SPEED RES write back
     * copy the device result to host pinned memory
     */
    cudaCheckError(cudaMemcpyAsync(speedRes_host_pinned, rx0_extended_fftRes_transpose, sizeof(Complex_t) * ChirpSize * extended_sample_size, cudaMemcpyDeviceToHost, speedStream));
    // cudaCheckError(cudaFreeAsync(rx0_extended_fft_input_device, speedStream));
    // cudaCheckError(cudaFreeAsync(rx0_extended_fftRes_transpose, speedStream));

    cudaCheckError(cudaEventRecord(speedEvt, speedStream));
}

void cudaMultiStreamAcceleration(short *input_host, Complex_t *base_frame_device,
                                 Complex_t *frame_reshaped_device, Complex_t *rx0_fft_input_device_dist,
                                 Complex_t *rx_fft_input_device_angle, Complex_t *frame_reshaped_device_angle,
                                 Complex_t *base_frame_device_angle, Complex_t *angle_weights_device,
                                 Complex_t *rx0_fft_input_device_angle, Complex_t *angle_matrix_device, Complex_t *angle_matrix_res_device,
                                 Complex_t *rx0_extended_fftRes_transpose, Complex_t *rx0_extended_fft_input_device,
                                 int size, int rx0_extended_size)
{
    Timer timer;
    double start = timer.elapsed();
    /**
     * create streams and event for syncrhonization and conccurent processing
     */
    cudaStream_t distStream;
    cudaStream_t *distStreamPtr = &distStream;

    cudaStream_t angleStream;
    cudaStream_t *angleStreamPtr = &angleStream;

    cudaStream_t speedStream;
    cudaStream_t *speedStreamPtr = &speedStream;

    cudaStream_t preProcStream;
    cudaStream_t *preProcStreamPtr = &preProcStream;

    cudaCheckError(cudaStreamCreate(&distStream));
    cudaCheckError(cudaStreamCreate(&angleStream));
    cudaCheckError(cudaStreamCreate(&speedStream));
    cudaCheckError(cudaStreamCreate(&preProcStream));

    cudaEvent_t preProcEvt;
    cudaEvent_t *preProcEvtPtr = &preProcEvt;

    cudaEvent_t angleEvt;
    cudaEvent_t *angleEvtPtr = &angleEvt;

    cudaEvent_t distEvt;
    cudaEvent_t *distEvtPtr = &distEvt;

    cudaEvent_t speedEvt;
    cudaEvent_t *speedEvtPtr = &speedEvt;

    cudaCheckError(cudaEventCreateWithFlags(&preProcEvt, cudaEventDisableTiming));
    cudaCheckError(cudaEventCreateWithFlags(&angleEvt, cudaEventDisableTiming));
    cudaCheckError(cudaEventCreateWithFlags(&distEvt, cudaEventDisableTiming));
    cudaCheckError(cudaEventCreateWithFlags(&speedEvt, cudaEventDisableTiming));

    Complex_t *rx0_device;
    cudaCheckError(cudaMalloc((void **)&rx0_device, sizeof(Complex_t) * rx0_extended_size));
    Complex_t *distRes_fft_host_pinned;
    cudaCheckError(cudaMallocHost((void **)&distRes_fft_host_pinned, sizeof(Complex_t) * rx0_extended_size));

    Complex_t *speedRes_host_pinned;
    cudaCheckError(cudaMallocHost((void **)&speedRes_host_pinned, sizeof(Complex_t) * rx0_extended_size));

    Complex_t *angleRes_host_pinned;
    int angle_sample_num = 180 / 0.1 + 1;
    cudaCheckError(cudaMallocHost((void **)&angleRes_host_pinned, sizeof(Complex_t) * angle_sample_num));
    Complex_t *base_frame_rx0_device;
    cudaCheckError(cudaMalloc((void **)&base_frame_rx0_device, sizeof(Complex_t) * SampleSize * ChirpSize));

    long memSize = sizeof(Complex_t) * rx0_extended_size * 3 + sizeof(Complex_t) * SampleSize * ChirpSize;
    // printf("allocated size before stream starts %.3f KiBytes\n", (double)(memSize / 1024));

    /**
     * launch pre processing stream
     */
    // launchPrePorc(input_host, base_frame_device, base_frame_rx0_device, rx0_device,
    //               frame_reshaped_device, size, rx0_extended_size, &preProcEvt, &preProcStream);
    double preProc = timer.elapsed();
    launchPrePorc(input_host, base_frame_device, base_frame_rx0_device, rx0_device,
                  frame_reshaped_device, size, rx0_extended_size, *preProcEvtPtr, *preProcStreamPtr);
    preProc = timer.elapsed() - preProc;
    // hold the host untill the preprocessing stage is finished
    cudaCheckError(cudaEventSynchronize(preProcEvt));
    /**
     * launch distance processing stream
     */
    double distProc = timer.elapsed();
    launchDistProc(*preProcEvtPtr, *distEvtPtr, *distStreamPtr, distRes_fft_host_pinned, rx0_device, rx0_fft_input_device_dist, rx0_extended_size);
    distProc = timer.elapsed() - distProc;
    /**
     * launch speed processing
     */
    double speedProc = timer.elapsed();
    launchSpeedProc(*preProcEvtPtr, *speedEvtPtr, *speedStreamPtr,
                    frame_reshaped_device, base_frame_rx0_device, speedRes_host_pinned,
                    rx0_extended_fftRes_transpose, rx0_extended_fft_input_device,
                    rx0_extended_size);
    speedProc = timer.elapsed();
    // only when distance is calculated the angle stream is able to launch
    cudaCheckError(cudaEventSynchronize(distEvt));
    /**
     * distance result host processing
     */
    cudaCheckError(cudaStreamSynchronize(distStream));
    int maxDistIdx = findAbsMax(distRes_fft_host_pinned, floor(0.4 * rx0_extended_size));
    maxDistIdx = maxDistIdx * (ChirpSize * SampleSize) / rx0_extended_size;
    double Fs_extend = fs * rx0_extended_size / (ChirpSize * SampleSize);

    double distance = lightSpeed * (((double)maxDistIdx / rx0_extended_size) * Fs_extend) / (2 * mu);
    /**
     * launch angle processing only when max distance index is found
     */
    double angleProc = timer.elapsed();
    launchAngleProc(*preProcEvtPtr, *distEvtPtr, *angleEvtPtr, *angleStreamPtr,
                    frame_reshaped_device, base_frame_device,
                    distRes_fft_host_pinned, angleRes_host_pinned,
                    rx_fft_input_device_angle, frame_reshaped_device_angle, base_frame_device_angle,
                    angle_weights_device, rx0_fft_input_device_angle, angle_matrix_device, angle_matrix_res_device,
                    rx0_extended_size, maxDistIdx);
    angleProc = timer.elapsed() - angleProc;
    /**
     * result host processing
     */

    cudaCheckError(cudaStreamSynchronize(speedStream));
    int maxSpeedIdx = findAbsMax(speedRes_host_pinned, ChirpSize * SampleSize) % ChirpSize;
    double fr = 1e6 / 64;
    double LAMDA = 3.0e08 / 77e9;
    double maxSpeed = ((maxSpeedIdx)*fr / ChirpSize - fr / 2) * LAMDA / 2;

    cudaCheckError(cudaStreamSynchronize(angleStream));
    int maxAngleIdx = findAbsMax(angleRes_host_pinned, angle_sample_num);
    double maxAngle = (maxAngleIdx - 900.0) / 10.0;
    double duration = timer.elapsed() - start;
    double fps = 1 / duration;

    multiStreamCnt++;

    if (multiStreamCnt == 89)
    {
        printf("distance %.3f m angle %.3f degree speed %.3f m/s\n", distance, maxAngle, maxSpeed);
        printf("total time %.3f ms fps %.3f \n", (1000 * duration), fps);
        printf("preProcessing time %.3f ms fps %.3f \n", (1000 * preProc), 1 / preProc);
        printf("distance time %.3f ms, fps %.3f\n", (1000 * distProc), 1 / distProc);
        printf("angle time %.3f ms, fps %.3f\n", (1000 * angleProc), 1 / angleProc);
        printf("speed time %.3f ms, fps %.3f\n", (1000 * speedProc), 1 / speedProc);
        printf("multiStreamCnt %d\n", multiStreamCnt);
    }
    cudaDeviceSynchronize();

    // wait for all stream finished

    cudaCheckError(cudaFreeHost(distRes_fft_host_pinned));
    cudaCheckError(cudaFreeHost(angleRes_host_pinned));
    cudaCheckError(cudaFreeHost(speedRes_host_pinned));

    cudaCheckError(cudaFree(rx0_device));
    cudaCheckError(cudaFree(base_frame_rx0_device));
    cudaCheckError(cudaEventDestroy(preProcEvt));
    cudaCheckError(cudaEventDestroy(distEvt));
    cudaCheckError(cudaEventDestroy(angleEvt));
    cudaCheckError(cudaEventDestroy(speedEvt));
}

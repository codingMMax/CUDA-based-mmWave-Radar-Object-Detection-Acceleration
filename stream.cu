#include "stream.cuh"

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
    //     printf("buf_complex[%d] real: %.5f imag:%.5f \n",i,buf_complex[i].real, buf_complex[i].imag);
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

__device__ int cudaFindAbsMax(Complex_t *ptr, int size)
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
 * Device function perform bit reverse and element swap for later fft
 * This function is doing the same as the kernel function, but is designed
 * to be called within a kernel before later fft.
 */
__device__ void bitReverseSwap_func(Complex_t *input, int size, int pow)
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
 * device function perform butterfly computation fft for input data.
 * This function is the same as butterflyFFT_kernel function, but
 * it this is designed to be called within a kernel function for better
 * overall parallelsim.
 */
__device__ void butterflyFFT_func(Complex_t *data, int size, int stage, int pow)
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
 * Wrapper function to luanch cuda kernels
 * @param input_host: data read from .bin file in short format, with length 'size' = 'SampleSize * ChirpSize * numRx * 2'.
 * @param base_frame_rx0_device: allocated base frame rx0 data space in device side, with length 'SampleSize * ChirpSize'.
 * @param frame_buffer_device: allocated frame reshape buffer sapce in device side, with length 'size'.
 * @param frame_reshaped_device: allocated reshaped frame data space in device side, with length 'size/2'.
 * @param frame_reshaped_rx0_device: allocated reshaped frame rx0 data space in device side, with length 'rx0_extended_size'.
 * @param size: int type indicates the total length of 'input_host'.
 * @param rx0_extended_size: int type indicates the length of 'rx0_extended_size'.
 * @return: double format calculated distance of moving object
 *
 */
// double cudaAcceleration(double &fftTime, double &preProcessingTime, double &findMaxTime, double &totalTime, short *input_host, Complex_t *base_frame_rx0_device, Complex_t *frame_buffer_device, Complex_t *frame_reshaped_device, Complex_t *frame_reshaped_rx0_device, int size, int rx0_extended_size)
double cudaAcceleration(double &angleTime, double &distTime, double &fftTime, double &preProcessingTime, double &findMaxTime, double &totalTime, short *input_host, Complex_t *base_frame_device, Complex_t *frame_buffer_device, Complex_t *frame_reshaped_device, Complex_t *frame_reshaped_rx0_device, int size, int rx0_extended_size)
{
    Timer timer;
    double start = timer.elapsed();
    int num_blocks_preProcessing = (THREADS_PER_BLOCK + size - 1) / THREADS_PER_BLOCK;
    /**Allocate memory space for current frame*/
    short *input_device;
    cudaCheckError(cudaMalloc((void **)&input_device, sizeof(short) * size));
    cudaCheckError(cudaMemcpy(input_device, input_host, sizeof(short) * size, cudaMemcpyHostToDevice));
    Complex_t *extended_rx0;
    cudaCheckError(cudaMalloc((void **)&extended_rx0, sizeof(Complex_t) * rx0_extended_size));
    Complex_t *rx0_fft_input_device;
    cudaCheckError(cudaMalloc((void **)&rx0_fft_input_device, sizeof(Complex_t) * rx0_extended_size));
    Complex_t *preProcessing_buffer;
    cudaCheckError(cudaMalloc((void **)&preProcessing_buffer, sizeof(Complex_t) * size / 2));

    short2complex_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(input_device, preProcessing_buffer, size);
    cudaDeviceSynchronize();

    // printf("buffer\n");
    // printComplexCUDA(preProcessing_buffer, 46662, 46668, SampleSize * ChirpSize * RxSize);

    num_blocks_preProcessing = (THREADS_PER_BLOCK + size / 2 - 1) / THREADS_PER_BLOCK;
    complexReshape_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(frame_reshaped_device, preProcessing_buffer, size / 2);
    cudaDeviceSynchronize();

    // printf("frame_reshaped_device\n");
    // printComplexCUDA(frame_reshaped_device, SampleSize * ChirpSize + 100, SampleSize * ChirpSize + 110, SampleSize * ChirpSize * RxSize);

    // printf("base_frame_device\n");
    // printComplexCUDA(base_frame_device, SampleSize * ChirpSize + 100, SampleSize * ChirpSize + 110, SampleSize * ChirpSize * RxSize);

    Complex_t *base_frame_rx0_device;
    cudaCheckError(cudaMalloc((void **)&base_frame_rx0_device, sizeof(Complex_t) * SampleSize * ChirpSize));
    cudaCheckError(cudaMemcpy(base_frame_rx0_device, base_frame_device, sizeof(Complex_t) * SampleSize * ChirpSize, cudaMemcpyDeviceToDevice));
    num_blocks_preProcessing = (THREADS_PER_BLOCK + rx0_extended_size - 1) / THREADS_PER_BLOCK;
    rxExtension_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(base_frame_rx0_device, rx0_fft_input_device, frame_reshaped_device, SampleSize * ChirpSize, rx0_extended_size);
    // printf("after reshaped kernel FFT input \n");
    // printComplexCUDA(rx0_fft_input_device, 100, 110, rx0_extended_size);

    double preProcessingEnd = timer.elapsed();
    double fftStart = preProcessingEnd;
    preProcessingTime += preProcessingEnd - start;

    int cnt = 1, pow = 0;
    while (cnt < rx0_extended_size)
    {
        cnt <<= 1;
        pow++;
    }
    bitReverseSwap_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(rx0_fft_input_device, rx0_extended_size, pow);
    cudaDeviceSynchronize();
    for (int i = 0; i < pow; i++)
    {
        butterflyFFT_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(rx0_fft_input_device, rx0_extended_size, i + 1, pow);
        cudaDeviceSynchronize();
    }

    double fftEnd = timer.elapsed();
    double findMaxStart = fftEnd;
    fftTime += fftEnd - fftStart;
    // printf("FFT res \n");
    // printComplexCUDA(rx0_fft_input_device, rx0_extended_size * 2 / 3, rx0_extended_size * 2 / 3 + 10, rx0_extended_size);

    double Fs_extend = fs * rx0_extended_size / (ChirpSize * SampleSize);

    Complex_t *rx0_fft_res_host = (Complex_t *)malloc(sizeof(Complex_t) * rx0_extended_size);
    cudaCheckError(cudaMemcpy(rx0_fft_res_host, rx0_fft_input_device, sizeof(Complex_t) * rx0_extended_size, cudaMemcpyDeviceToHost));
    int maxIdx = findAbsMax(rx0_fft_res_host, floor(0.4 * rx0_extended_size));
    int maxDisIdx = maxIdx * (ChirpSize * SampleSize) / rx0_extended_size;

    double maxDis = lightSpeed * (((double)maxDisIdx / rx0_extended_size) * Fs_extend) / (2 * mu);

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
        cudaDeviceSynchronize();
        for (int stage = 0; stage < pow; stage++)
        {
            butterflyFFT_kernel<<<num_blocks_preProcessing, THREADS_PER_BLOCK>>>(rx_fft_input_device_ptr, rx0_extended_size, stage + 1, pow);
            cudaDeviceSynchronize();
        }
        fftTime += (timer.elapsed() - angleFFTBegin);
    }
    // above operations are verified
    // assign values to angle weigths
    Complex_t *angle_weights_device;
    cudaCheckError(cudaMalloc((void **)&angle_weights_device, sizeof(Complex_t) * RxSize));
    angleWeightInit_kernel<<<1, RxSize>>>(angle_weights_device, rx0_fft_input_device, rx_fft_input_device, maxAngleIdx, rx0_extended_size);
    // printComplexCUDA(angle_weights_device, 0, RxSize, RxSize);
    // above operations are verified
    // Stage4 MMM: Angle Matrix X angle_weights
    Complex_t *angle_matrix_device;
    int angle_sample_num = 180 / 0.1 + 1;
    cudaCheckError(cudaMalloc((void **)&angle_matrix_device, RxSize * angle_sample_num * sizeof(Complex_t)));
    num_blocks_angle = (RxSize * angle_sample_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaDeviceSynchronize();

    angleMatrixInit_kernel<<<num_blocks_angle, THREADS_PER_BLOCK>>>(angle_matrix_device, RxSize * angle_sample_num);
    cudaDeviceSynchronize();
    // printComplexCUDA(angle_matrix_device, angle_sample_num * 2, angle_sample_num * 2 + 10, RxSize * angle_sample_num);
    num_blocks_angle = (angle_sample_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    Complex_t *angle_matrix_res_device;
    Complex_t *angle_matrix_res_host;
    cudaCheckError(cudaMalloc((void **)&angle_matrix_res_device, sizeof(Complex_t) * angle_sample_num));
    angle_matrix_res_host = (Complex_t *)malloc(sizeof(Complex_t) * angle_sample_num);
    angleMatrixMul_kernel<<<num_blocks_angle, THREADS_PER_BLOCK>>>(angle_matrix_device, angle_weights_device, angle_matrix_res_device, angle_sample_num);
    cudaDeviceSynchronize();
    cudaCheckError(cudaMemcpy(angle_matrix_res_host, angle_matrix_res_device, sizeof(Complex_t) * angle_sample_num, cudaMemcpyDeviceToHost));
    maxAngleIdx = findAbsMax(angle_matrix_res_host, angle_sample_num);
    // printComplexCUDA(angle_matrix_res_device, 1100, 1110, angle_sample_num);
    // above operations are verified
    double maxAgl = (maxAngleIdx - 900.0) / 10.0;
    printf("maxAgl %.9f\n",maxAgl);

    angleTime += (timer.elapsed() - angleBegin);

    totalTime += (timer.elapsed() - start);

    free(rx0_fft_res_host);
    free(angle_matrix_res_host);

    cudaCheckError(cudaFree(rx_fft_input_device));
    cudaCheckError(cudaFree(base_frame_rx0_device));
    cudaCheckError(cudaFree(angle_weights_device));
    cudaCheckError(cudaFree(angle_matrix_device));
    cudaCheckError(cudaFree(angle_matrix_res_device));
    cudaCheckError(cudaFree(input_device));
    cudaCheckError(cudaFree(extended_rx0));
    cudaCheckError(cudaFree(rx0_fft_input_device));
    cudaCheckError(cudaFree(preProcessing_buffer));

    return maxDis;
}

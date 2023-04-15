#include "acceleration.h"
#include <time.h>
#define SampleSize 100 // the sample number in a chirp, suggesting it should be the power of 2
#define ChirpSize 128  // the chirp number in a frame, suggesting it should be the the power of 2
#define FrameSize 90   // the frame number
#define RxSize 4       // the rx size, which is usually 4
#define c 3.0e8        // the speed of light
#define pi 3.14125     // pi

double F0 = 77e9;            // the initial frequency
double mu = 5.987e12;        // FM slope
double samplePerChirp = 100; // the sample number in a chirp, suggesting it should be the power of 2
double Fs = 2.0e6;           // sampling frequency
double numChirp = 128;       // the chirp number in a frame, suggesting it should be the the power of 2
double frameNum = 90;        // the frame number
double Tr = 64e-6;           // the interval of the chirp
double fr = 1 / Tr;          // chirp repeating frequency,
double lamda = c / F0;       // lamda of the initial frequency
double d = 0.5 * lamda;      // rx_wire array distance. When it is equal to the half of the wavelength, the
// maximum unambiguous Angle can reach -90° to +90°
int numTx = 1;
int numRx = 4; // the rx size, which is usually 4

Complex_t GetComplex_t(double r, double i)
{
    Complex_t temp;
    temp.real = r;
    temp.imag = i;
    return temp;
}

Complex_t Complex_t_ADD(Complex_t a, Complex_t b)
{
    Complex_t temp;
    temp.real = a.real + b.real;
    temp.imag = a.imag + b.imag;
    return temp;
}

Complex_t Complex_t_SUB(Complex_t a, Complex_t b)
{
    Complex_t temp;
    temp.real = a.real - b.real;
    temp.imag = a.imag - b.imag;
    return temp;
}

Complex_t Complex_t_MUL(Complex_t a, Complex_t b)
{
    Complex_t temp;
    temp.real = a.real * b.real - a.imag * b.imag;
    temp.imag = a.real * b.imag + a.imag * b.real;
    return temp;
}

double Complex_t_mol(Complex_t *a)
{
    return sqrt(a->real * a->real + a->imag * a->imag);
}

int reverseBits(int num, int bits)
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
void butterfly_fft(int size, Complex_t input[])
{
    // Bit reversal
    int bits = log2(size);
    for (int i = 0; i < size; i++)
    {
        int j = reverseBits(i, bits);
        if (j > i)
        {
            Complex_t temp = input[i];
            input[i] = input[j];
            input[j] = temp;
        }
    }

    // Cooley-Tukey radix-2 DIT FFT
    for (int step = 2; step <= size; step <<= 1)
    {
        double theta = -2 * M_PI / step;
        Complex_t twiddle = {cos(theta), sin(theta)};
        for (int k = 0; k < size; k += step)
        {
            Complex_t omega = {1.0, 0.0};
            for (int j = 0; j < step / 2; j++)
            {
                Complex_t wn = Complex_t_MUL(omega, input[k + j + step / 2]);
                input[k + j + step / 2] = Complex_t_SUB(input[k + j], wn);
                input[k + j] = Complex_t_ADD(input[k + j], wn);
                omega = Complex_t_MUL(omega, twiddle);
            }
        }
    }
}

// get the extended size
int nextPow2(int n)
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

/// C read the binary file size
int getBinSize(char *path)
{
    int size = 0;
    FILE *fp = fopen(path, "rb");
    if (fp)
    {
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fclose(fp);
    }
    return size;
}

// C read the bin data in size of short
void readBin(char *path, short *buf, int size)
{
    FILE *infile;
    if ((infile = fopen(path, "rb")) == NULL)
    {
        printf("\nCan not open the path: %s \n", path);
    }
    fread(buf, sizeof(short), size, infile);
    fclose(infile);
}

// reshape the input bin file
// Input: *OriginalArray: the input of short bin file, size: the real size in form of short
// Output: *Reshape: reshape the input in form of complex
void ReshapeComplex_t(short *OriginalArray, Complex_t *Reshape, int size)
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

// find the max of the abs of the complex array
int FindAbsMax(Complex_t *ptr, int size)
{
    int maxidx = 0;
    double maxval = 0;
    double absval;
    for (int i = 0; i < size; i++)
    {
        absval = Complex_t_mol((ptr + i));
        if (absval > maxval)
        {
            maxval = absval;
            maxidx = i;
        }
    }
    return maxidx;
}

void printComplex(Complex_t in)
{
    printf("img %.6f real %.6f \n", in.imag, in.real);
}

void cpuTiming()
{
    double fftTime;
    // reshape and extension time
    double preProcessTime;
    double findMaxTime;

    Timer timer;
    double totalBegin = timer.elapsed();
    // clock_t totalBegin = clock();

    char filepath[] = "./fhy_direct.bin";
    // char filepath[] = "./new_sample.bin";
    int NumDataPerFrame = ChirpSize * SampleSize * numRx * 2;
    int BytePerFrame = NumDataPerFrame * sizeof(short);
    FILE *fp = fopen(filepath, "rb");

    if (fp == NULL)
    {
        printf("unable to read the specified file\n");
        return;
    }

    short *inputData = (short *)malloc(BytePerFrame);
    int size = 0;
    double cpuRes[FrameSize];

    int numFrameRead = 0;

    size = (int)fread(inputData, sizeof(short), NumDataPerFrame, fp);
    /**
     * 2. Reshape the data
     */
    // printf("Reading the baseFrame\n");
    Complex_t *baseFrameReshaped = (Complex_t *)malloc(size * sizeof(Complex_t) / 2);
    Complex_t *baseFrameRx0 = (Complex_t *)malloc(ChirpSize * SampleSize * sizeof(Complex_t));
    ReshapeComplex_t(inputData, baseFrameReshaped, size);
    memmove(baseFrameRx0, baseFrameReshaped, ChirpSize * SampleSize * sizeof(Complex_t));

    // printf("baseFrame processing finished\n");
    int extendSize = nextPow2(ChirpSize * SampleSize);

    Complex_t *frameDataReshaped = (Complex_t *)malloc(size * sizeof(Complex_t) / 2);
    Complex_t *frameDataRx0 = (Complex_t *)malloc(extendSize * sizeof(Complex_t));
    // allocate the memory for fft data
    Complex_t *fftRes = (Complex_t *)malloc(nextPow2(ChirpSize * SampleSize) * sizeof(Complex_t));
    Complex_t *fftInput = (Complex_t *)malloc(extendSize * sizeof(Complex_t));

    while ((size = (int)fread(inputData, sizeof(short), NumDataPerFrame, fp)) > 0)
    {

        // printf("Reading next frame from file\n");
        numFrameRead++;
        /**
         * input data into cuda devices
         */
        // printf("CUDA Processing elements\n");

        // printf("Reshape next frame\n");
        double preProcessBegin = timer.elapsed();
        ReshapeComplex_t(inputData, frameDataReshaped, size);

        memmove(frameDataRx0, frameDataReshaped, ChirpSize * SampleSize * sizeof(Complex_t));
        // extend the frame data
        for (int i = 0; i < SampleSize * ChirpSize; i++)
        {
            fftInput[i] = Complex_t_SUB(frameDataRx0[i], baseFrameRx0[i]);
        }
        for (int i = SampleSize * ChirpSize; i < extendSize; i++)
        {
            fftInput[i].real = 0;
            fftInput[i].imag = 0;
        }
        double preProcessEnd = timer.elapsed();

        preProcessTime += (preProcessEnd - preProcessBegin);

        double fftBegin = timer.elapsed();
        memmove(fftRes, fftInput, extendSize * sizeof(Complex_t));
        butterfly_fft(extendSize, fftRes);
        double fftEnd = timer.elapsed();
        fftTime += (fftEnd - fftBegin);

        // for(int i = extendSize * 2 / 3; i < extendSize * 2 / 3 + 10; i++){
        //     printf("CPU fftRes[%d] real %.5f imag %.5f \n", i,fftRes[i].real, fftRes[i].imag);
        // }

        double findMaxBegin = timer.elapsed();
        double Fs_extend = Fs * extendSize / (ChirpSize * SampleSize);
        int maxDisIdx = FindAbsMax(fftRes, floor(0.4 * extendSize)) * (ChirpSize * SampleSize) / extendSize;
        double maxDis = c * (((double)maxDisIdx / extendSize) * Fs_extend) / (2 * mu);
        double findMaxEnd = timer.elapsed();

        findMaxTime += (findMaxEnd - findMaxBegin);

        // printf("Finding maxDisIdx %d maxDis %.5f\n", maxDisIdx, maxDis);
        cpuRes[numFrameRead] = maxDis;
    }

    free(fftRes);
    free(fftInput);
    free(frameDataReshaped);
    free(frameDataRx0);

    free(baseFrameReshaped);
    free(baseFrameRx0);
    free(inputData);
    fclose(fp);

    double totalEnd = timer.elapsed();
    // clock_t totalEnd = clock();
    double totalTime = totalEnd - totalBegin;

    // double totalTime = (double)(totalEnd - totalBegin) / CLOCKS_PER_SEC ;

    printf("Total Time for %d frames %.5f ms averaged %.5f FPS \n", numFrameRead, 1000.0 * totalTime, (double)numFrameRead / totalTime);
    printf("Total FFT time %.5f ms averaged %.5f ms/frame \n", 1000.0 * fftTime, 1000.0 * fftTime / (double)numFrameRead);
    printf("Total Reshape + Extension time %.5f ms averaged %.5f / frame\n", 1000.0 * preProcessTime, 1000.0 * preProcessTime / (double)numFrameRead);
    printf("Total findMax time %.5f ms averaged %.5f ms/frame\n", 1000.0 * findMaxTime, 1000.0 * findMaxTime / (double)numFrameRead);
}

void cudaTiming()
{
    Timer timer;
    double cudaTime = timer.elapsed();

    char filepath[] = "./fhy_direct.bin";
    // char filepath[] = "./new_sample.bin";
    int NumDataPerFrame = ChirpSize * SampleSize * numRx * 2;
    int BytePerFrame = NumDataPerFrame * sizeof(short);
    FILE *fp = fopen(filepath, "rb");

    if (fp == NULL)
    {
        printf("unable to read the specified file\n");
        return;
    }

    short *inputData = (short *)malloc(BytePerFrame);
    int size = 0;
    double cudaRes[FrameSize];

    int numFrameRead = 0;

    size = (int)fread(inputData, sizeof(short), NumDataPerFrame, fp);
    /**
     * 2. Reshape the data
     */
    // printf("Reading the baseFrame\n");
    Complex_t *baseFrameReshaped = (Complex_t *)malloc(size * sizeof(Complex_t) / 2);
    Complex_t *baseFrameRx0 = (Complex_t *)malloc(ChirpSize * SampleSize * sizeof(Complex_t));
    ReshapeComplex_t(inputData, baseFrameReshaped, size);
    memmove(baseFrameRx0, baseFrameReshaped, ChirpSize * SampleSize * sizeof(Complex_t));

    // printf("baseFrame processing finished\n");
    int extendSize = nextPow2(ChirpSize * SampleSize);

    Complex_t *frameDataReshaped = (Complex_t *)malloc(size * sizeof(Complex_t) / 2);
    Complex_t *frameDataRx0 = (Complex_t *)malloc(extendSize * sizeof(Complex_t));
    double fftTime = 0, preProcessTime = 0, findMaxTime = 0, totalTime = 0;

    while ((size = (int)fread(inputData, sizeof(short), NumDataPerFrame, fp)) > 0)
    {
        numFrameRead++;
        cudaRes[numFrameRead] = cudaProcessing(inputData, baseFrameRx0, size, &fftTime, &preProcessTime, &findMaxTime, &totalTime);
    }

    free(frameDataReshaped);
    free(frameDataRx0);
    free(baseFrameReshaped);
    free(baseFrameRx0);
    free(inputData);

    fclose(fp);

    cudaTime = timer.elapsed() - cudaTime;

    printf("cuda totalTime %.5f ms average %.5f ms/frame cudaFPS %.5f FPS\n", 1000.0 * cudaTime, (1000.0 * cudaTime) / (double)numFrameRead, 1 / cudaTime * numFrameRead);
    printf("cuda FFT time %.5f ms average %.5f ms/frame\n", 1000.0 * fftTime, 1000.0 * fftTime / (double)numFrameRead);
    printf("cuda preProcesstime time %.5f ms average %.5f ms/frame\n", 1000.0 * preProcessTime, 1000.0 * preProcessTime / (double)numFrameRead);
    printf("cuda findMaxTime time %.5f ms average %.5f ms/frame\n", 1000.0 * findMaxTime, 1000.0 * findMaxTime / (double)numFrameRead);
    printf("cuda inner time %.5f ms average %.5f ms/frame cuda inner FPS %.5f FPS\n", 1000.0 * totalTime, 1000.0 * totalTime / (double)numFrameRead, 1 / totalTime * numFrameRead);
}

int main()
{
    /**
     * Operation in CPU side:
     * 1. Read data from file
     * 2. Reshape the data
     * 3. Extend the data
     * 4. Feed the data into CUDA
     * 5. Running local test for verification
     * 6. Accept the data from CUDA
     * 7. Verify the result
     */

    // for (int i = 1; i < numFrameRead; i++)
    // {
    //     if (abs(cudaRes[i] - cpuRes[i]) >= 1e-5)
    //     {
    //         printf("CUDA result verification failed at frame %d\n", i);
    //         printf("Ref Res %.6f CUDA res %.6f\n", cpuRes[i], cudaRes[i]);
    //         break;
    //     }
    //     // printf("frame[%d] Ref Res %.6f CUDA res %.6f\n", i, cpuRes[i], cudaRes[i]);
    // }
    printf("CPU Timing\n");
    cpuTiming();
    printf("CUDA Timing\n");
    cudaTiming();

    // char filepath[] = "./new_sample.bin";
    // FILE *fp;
    // int size;
    // if ((fp = fopen(filepath, "rb+")) == NULL)
    // {
    //     printf("\nCan not open the path: %s \n", filepath);
    //     return -1;
    // }
    // else
    // {
    //     fseek(fp, 0, SEEK_END);
    //     size = ftell(fp);
    // }
    // short *buf = (short *)malloc(sizeof(short) * size);
    // fread(buf, sizeof(short), size, fp);
    // fclose(fp);

    // fp = fopen("./new_sample.bin","ab+");
    // fwrite(buf, sizeof(short), size, fp);
    // fwrite(buf, sizeof(short), size, fp);

    // fclose(fp);

    return 0;
}

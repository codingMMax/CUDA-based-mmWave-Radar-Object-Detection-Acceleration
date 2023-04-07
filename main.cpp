#include <stdio.h>
#include <math.h>
// #include <iostream>
// #include <cmath>
#include <stdlib.h>
#include <string.h>
// #include<bits/stdc++.h>
//#include "FFT_C.cpp"

// using namespace std;

#define SampleSize 100              // the sample number in a chirp, suggesting it should be the power of 2
#define ChirpSize 128               // the chirp number in a frame, suggesting it should be the the power of 2
#define FrameSize  90             // the frame number
#define RxSize 4                    // the rx size, which is usually 4
#define c 3.0e8                     // the speed of light
#define pi 3.14125                  // pi


double F0 = 77e9;                      // the initial frequency
double mu = 5.987e12;                  // FM slope
double samplePerChirp = 100;             // the sample number in a chirp, suggesting it should be the power of 2
double Fs = 2.0e6;                     // sampling frequency
double numChirp = 128;                // the chirp number in a frame, suggesting it should be the the power of 2
double frameNum = 90;                  // the frame number
double Tr = 64e-6;                     // the interval of the chirp
double fr = 1 / Tr;                     // chirp repeating frequency,
double lamda = c / F0;                   // lamda of the initial frequency
double d = 0.5 * lamda;                  // rx_wire array distance. When it is equal to the half of the wavelength, the
// maximum unambiguous Angle can reach -90° to +90°
int numTx = 1;
int numRx = 4;                      // the rx size, which is usually 4


// complex struct and complex algorithm
struct Complex_t {
    double real, imag;
};

Complex_t GetComplex_t(double r, double i) {
    Complex_t temp;
    temp.real = r;
    temp.imag = i;
    return temp;
}

Complex_t Complex_t_ADD(Complex_t a, Complex_t b) {
    Complex_t temp;
    temp.real = a.real + b.real;
    temp.imag = a.imag + b.imag;
    return temp;
}

Complex_t Complex_t_SUB(Complex_t a, Complex_t b) {
    Complex_t temp;
    temp.real = a.real - b.real;
    temp.imag = a.imag - b.imag;
    return temp;
}

Complex_t Complex_t_MUL(Complex_t a, Complex_t b) {
    Complex_t temp;
    temp.real = a.real * b.real - a.imag * b.imag;
    temp.imag = a.real * b.imag + a.imag * b.real;
    return temp;
}

Complex_t Complex_t_CDiv(Complex_t a, double b) {
    Complex_t temp;
    temp.real = a.real / b;
    temp.imag = a.imag / b;
    return temp;
}

void Complex_t_MetricMUL(Complex_t *M_res, Complex_t *M_A, Complex_t *M_B, int sizea, int sizeb, int sizec) {
    // M_A = a*b
    // M_B = b*c
    // M_res = a*c
    for (int i = 0; i < sizea; i++) {
        for (int j = 0; j < sizeb; j++) {
            for (int k = 0; k < sizec; k++) {
                *(M_res + i * sizea + k) = Complex_t_ADD(
                        Complex_t_MUL((*(M_A + i * sizea + j)), (*(M_B + j * sizeb + k))),
                        *(M_res + i * sizea + k));
            }
        }
    }
}

double Complex_t_mol(Complex_t *a) {
    return sqrt(a->real * a->real + a->imag * a->imag);
}

// FFT recursive version
// Input : a pointer to a Complex_t Array, the size of the array
// Output: rewrite the output to x[]
void FFT_recursive(Complex_t x[], int len) {
    Complex_t *odd, *even;
    Complex_t t;
    if (len == 1) return;
    odd = (Complex_t *) malloc(sizeof(Complex_t) * len / 2);
    even = (Complex_t *) malloc(sizeof(Complex_t) * len / 2);

    for (int i = 0; i < len / 2; i++) {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    FFT_recursive(odd, len / 2);
    FFT_recursive(even, len / 2);

    for (int i = 0; i < len / 2; i++) {
        t = Complex_t_MUL(GetComplex_t(cos(2.0 * pi * i / len), -sin(2.0 * pi * i / len)), odd[i]);
        x[i] = Complex_t_ADD(even[i], t);
        x[i + len / 2] = Complex_t_SUB(even[i], t);
    }

    free(odd);
    free(even);
}


// FFT nonrecursive version
// Input : a pointer to a Complex_t Array, the size of the array
// Output: rewrite the output to x[]
void FFT_nonrecursive(Complex_t x[], int len) {
    int temp = 1, l = 0;
    int *r = (int *) malloc(sizeof(int) * len);
    Complex_t t;
    //FFT index reverse，get the new index，l=log2(len)
    while (temp < len) temp <<= 1, l++;
    for (int i = 0; i < len; i++) r[i] = (r[i >> 1] >> 1) | ((i & 1) << (l - 1));
    //swap according to the index
    for (int i = 0; i < len; i++)
        if (i < r[i]) {
            t = x[i];
            x[i] = x[r[i]];
            x[r[i]] = t;
        }
    for (int mid = 1; mid < len; mid <<= 1) {
        Complex_t Wn = GetComplex_t(cos(pi / mid),
                                    -sin(pi / mid)); /*drop the "-" sin，then divided by len to get the IFFT*/
        for (int R = mid << 1, j = 0; j < len; j += R) {
            Complex_t w = GetComplex_t(1, 0);
            for (int k = 0; k < mid; k++, w = Complex_t_MUL(w, Wn)) {
                Complex_t a = x[j + k], b = Complex_t_MUL(w, x[j + mid + k]);
                x[j + k] = Complex_t_ADD(a, b);
                x[j + mid + k] = Complex_t_SUB(a, b);
            }
        }
    }
    free(r);
}


// get the extended size
int getextendsize(int size) {
    int pow2, powres;
    powres = 1;
    for (pow2 = 1; powres < size; pow2++) {
        powres *= 2;
    }
    return powres;
}

// FFTextend: extend and execute the FFT(recursive or nonrecursive)
// Input : *Data: a pointer to a Complex_t Array, size: the size of the array
// Output: *Res: FFT output of the extended input
void FFTextend(Complex_t *Data, Complex_t *Res, int size) {
    int powres, i;
    powres = getextendsize(size);

    Complex_t *Data_extend = (Complex_t *) malloc(powres * sizeof(Complex_t));
    memmove(Data_extend, Data, size * sizeof(Complex_t));
    for (i = size; i < powres; i++) {
        (Data_extend + i)->real = 0;
        (Data_extend + i)->imag = 0;
    }

    FFT_nonrecursive(Data_extend, powres);
    // FFT_recursive(Data_extend, powres);

    memmove(Data, Data_extend, size * sizeof(Complex_t));
    memmove(Res, Data_extend, powres * sizeof(Complex_t));
    free(Data_extend);
}


/// C read the binary file size
int getBinSize(char *path) {
    int size = 0;
    FILE *fp = fopen(path, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fclose(fp);
    }
    return size;
}

// C read the bin data in size of short
void readBin(char *path, short *buf, int size) {
    FILE *infile;
    if ((infile = fopen(path, "rb")) == NULL) {
        printf("\nCan not open the path: %s \n", path);
    }
    fread(buf, sizeof(short), size, infile);
    fclose(infile);
}

// C write the bin data in size of short
void writeBin(char *path, short *buf, int size) {
    FILE *outfile;
    if ((outfile = fopen(path, "wb")) == NULL) {
        printf("\nCan not open the path: %s \n", path);
    }
    fwrite(buf, sizeof(short), size, outfile);
    fclose(outfile);
}

// C write the bin data in size of Complex_t
void writeComplex_tBin(Complex_t *buf, int size) {
    char saveFilePath_real[] = "./res_reshape_real.bin";
    char saveFilePath_imag[] = "./res_reshape_imag.bin";

    short *real = (short *) malloc(size * sizeof(short));
    short *imag = (short *) malloc(size * sizeof(short));

    Complex_t *ptr = buf;
    printf("\nwrite size = %d\n ", size);
    for (int i = 0; i < size; i++) {
        *(real + i) = (short) (ptr->real);
        *(imag + i) = (short) (ptr->imag);
        ptr++;
    }

    writeBin(saveFilePath_imag, imag, size);
    int size_imag = getBinSize(saveFilePath_imag);
    printf("write2 finished, size = %d short\n", size_imag);

    writeBin(saveFilePath_real, real, size);
    int size_real = getBinSize(saveFilePath_real);
    printf("write1 finished, size = %d short\n", size_real);
}

// reshape the input bin file
// Input: *OriginalArray: the input of short bin file, size: the real size in form of short
// Output: *Reshape: reshape the input in form of complex
void ReshapeComplex_t(short *OriginalArray, Complex_t *Reshape, int size) {
    int i, j, k, l;
    int cnt = 0;
    Complex_t *buf_complex = (Complex_t *) malloc(size * sizeof(Complex_t) / 2);
    short *ptr;
    Complex_t *complex_ptr = buf_complex;
    // reshape into 2 form of complex
    for (i = 0; i < size; i += 4) {
        ptr = OriginalArray + i;
        complex_ptr->real = (double) *(ptr);
        complex_ptr->imag = (double) *(ptr + 2);
        complex_ptr++;
        complex_ptr->real = (double) *(ptr + 1);
        complex_ptr->imag = (double) *(ptr + 3);
        complex_ptr++;
    }
    Complex_t *Reshape_ptr;
    // change the sequence of the array, reshape it in form of Rx instead of frame
    for (i = 0; i < RxSize; i++) {
        for (j = 0; j < FrameSize * ChirpSize; j++) {
            for (k = 0; k < SampleSize; k++) {
                Reshape_ptr = (Reshape + i * FrameSize * ChirpSize * SampleSize + j * SampleSize + k);
                complex_ptr = (buf_complex + j * RxSize * SampleSize + i * SampleSize + k);
                Reshape_ptr->real = complex_ptr->real;
                Reshape_ptr->imag = complex_ptr->imag;
            }
        }
    }
    free(buf_complex);
    return;
}

// read the file and call the "ReshapeComplex_t" to reshape
int readandreshape(char *filepath, Complex_t *Data_reshape) {
    // ----------------------read size------------------------------
    char filePath[] = "./fhy_direct.bin";
    int bytesize = getBinSize(filePath);
    int size = bytesize / sizeof(short);
    // ----------------------read int16 ------------------------------
    short *buf = (short *) malloc(size * sizeof(short));
    readBin(filePath, buf, size);
    // ----------------------reshape ------------------------------
    short *buf_ptr = (short *) buf;
    short *buf_reshape_real = (short *) malloc(size * sizeof(short) / 2);
    short *buf_reshape_imag = (short *) malloc(size * sizeof(short) / 2);
    Complex_t *buf_reshape = (Complex_t *) malloc(size * sizeof(Complex_t) / 2);
    ReshapeComplex_t(buf_ptr, buf_reshape, size);
    return size / 2;
}

// find the max
int FindMax(double *ptr, int size) {
    int maxidx = 0;
    int maxval = 0;
    for (int i = 0; i < size; i++) {
        if (*(ptr) > maxval) {
            maxval = *(ptr);
            maxidx = i;
        }
        ptr++;
    }
    return maxidx;
}

// find the max of the abs of the complex array
int FindAbsMax(Complex_t *ptr, int size) {
    int maxidx = 0;
    double maxval = 0;
    double absval;
    for (int i = 0; i < size; i++) {
        absval = Complex_t_mol((ptr + i));
        if (absval > maxval) {
            maxval = absval;
            maxidx = i;
        }
    }
    return maxidx;
}

// Detect the distance ,
// Input : the bin size file
// Output: the dis array
void DetectDis(char *filepath, double *Dis) {
    // ----------------------read size------------------------------
    // char filePath[] = "./fhy_direct.bin";
    int bytesize = getBinSize(filepath);
    int size = bytesize / sizeof(short);
//    printf("byteSize %d numOfElements %d\n", bytesize, size);
    // ----------------------read int16 ------------------------------
    short *Data = (short *) malloc(size * sizeof(short));
    readBin(filepath, Data, size);
    // ----------------------reshape ------------------------------
    Complex_t *Data_reshape = (Complex_t *) malloc(size * sizeof(Complex_t) / 2);
    ReshapeComplex_t(Data, Data_reshape, size);
    size = size / 2;

    // get data_rx0
    int rx0_size = size / 4;
    Complex_t *Data_rx0_reshape = (Complex_t *) malloc(FrameSize * ChirpSize * SampleSize * sizeof(Complex_t));
    memmove(Data_rx0_reshape, Data_reshape, FrameSize * ChirpSize * SampleSize * sizeof(Complex_t));

    // malloc for the data
    Complex_t *Data_fft = (Complex_t *) malloc(getextendsize(ChirpSize * SampleSize) * sizeof(Complex_t));
//    Complex_t *DataFrm_rx = (Complex_t *) malloc(FrameSize * ChirpSize * SampleSize * sizeof(Complex_t));
    Complex_t *DataFrm_rx = (Complex_t *) malloc(1 * ChirpSize * SampleSize * sizeof(Complex_t));
// double *Dis = (double *)malloc((FrameSize) * sizeof(double));

    // caculate the position and speed
    int frm, chp, smp;
    int testfrm = 0;
    int numFrm = FrameSize;
    for (frm = testfrm; frm < testfrm + numFrm; frm++) {
        memmove(DataFrm_rx, Data_rx0_reshape + frm * ChirpSize * SampleSize,
                ChirpSize * SampleSize * sizeof(Complex_t));
        for (int i = 0; i < ChirpSize * SampleSize; i++) {
            *(DataFrm_rx + i) = Complex_t_SUB(*(DataFrm_rx + i), *(Data_rx0_reshape + i));
        }
        //---------------------------------------------------distance----------------------------------------------------

        FFTextend(DataFrm_rx, Data_fft, ChirpSize * SampleSize);

        int extendsize = getextendsize(ChirpSize * SampleSize);
        double Fs_extend = Fs * extendsize / (ChirpSize * SampleSize);

        int maxDisIdx = FindAbsMax(Data_fft, floor(0.4 * extendsize)) * (ChirpSize * SampleSize) / extendsize;
        double maxDis = c * (((double) maxDisIdx / extendsize) * Fs_extend) / (2 * mu);
        *(Dis + frm - 30) = maxDis;

        printf("\nfrm = %d\n", frm);
        printf("maxDisIdx = %d maxDis = %lf\n", maxDisIdx, maxDis);
    }
    free(DataFrm_rx);
    free(Data_fft);
    free(Data_rx0_reshape);
    free(Data_reshape);
    free(Data);
    // free(Dis);
}


void printComplex(Complex_t in) {
    printf("img %.6f real %.6f \n", in.imag, in.real);
}

void test() {
    char filepath[] = "./fhy_direct.bin";
    // read the file
    int NumDataPerFrame = ChirpSize * SampleSize * numRx * 2;
    int BytePerFrame = NumDataPerFrame * sizeof(short);
    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL) {
        printf("unable to read the specified file\n");
        return;
    }
    auto inputData = (short *) malloc(BytePerFrame);
    int size = 0;
//    double *distance = (double *) malloc(FrameSize * sizeof(double));
    double distance[FrameSize];
    int numFrameRead = 0;

    // read the file for each frame
    // read the first frame for baseline calibration
    size = (int) fread(inputData, sizeof(short), NumDataPerFrame, fp);
    // ------------- reshape read data ------------------
    Complex_t *baseFrameReshaped = (Complex_t *) malloc(size * sizeof(Complex_t) / 2);
    Complex_t *baseFrameRx0 = (Complex_t *) malloc(ChirpSize * SampleSize * sizeof(Complex_t));

    ReshapeComplex_t(inputData, baseFrameReshaped, size);
    // ------------ extract rx0 data frame ----------------------
    // allocate the base frame data for calibration in later frames
    memmove(baseFrameRx0, baseFrameReshaped, ChirpSize * SampleSize * sizeof(Complex_t));
    printf("before enter the processing baseFrameRx0\n");
    for(int i = 0; i < 5; i ++ ){
        printComplex(baseFrameRx0[i]);
    }
//    numFrameRead++;
    // allocate memory for later frame
    Complex_t *frameDataReshaped = (Complex_t *) malloc(size * sizeof(Complex_t) / 2);
    Complex_t *frameDataRx0 = (Complex_t *) malloc(ChirpSize * SampleSize * sizeof(Complex_t));
    // allocate the memory for fft data
    Complex_t *fftRes = (Complex_t *) malloc(getextendsize(ChirpSize * SampleSize) * sizeof(Complex_t));
    Complex_t *frameRxData = (Complex_t *) malloc(ChirpSize * SampleSize * sizeof(Complex_t));

    // read the input data in each frame
    while ((size = (int)fread(inputData, sizeof(short), NumDataPerFrame, fp)) > 0) {
        numFrameRead++;

        // read 102400 num of elements each time
//        printf("numFrameRead %d read size = %lu\n", numFrameRead, size);
        // ------------------ reshape the read data -------------------
        // ------------------ extract Rx0 data frame -------------------
        ReshapeComplex_t(inputData, frameDataReshaped, size);
        memmove(frameDataRx0, frameDataReshaped, ChirpSize * SampleSize * sizeof(Complex_t));
        // allocate fft data
//        memmove(frameRxData,frameDataComplete,)

//        printf("frame %d: subtract the base frame data for calibration\n", numFrameRead);
        for (int i = 0; i < SampleSize * ChirpSize; i++) {
            frameRxData[i] = Complex_t_SUB(frameDataRx0[i], baseFrameRx0[i]);
        }

        FFTextend(frameRxData, fftRes, ChirpSize * SampleSize);

        int extendSize = getextendsize(ChirpSize * SampleSize);

        double Fs_extend = Fs * extendSize / (ChirpSize * SampleSize);

        int maxDisIdx = FindAbsMax(fftRes, floor(0.4 * extendSize)) * (ChirpSize * SampleSize) / extendSize;

        if(numFrameRead == 1){
//            printf("found maxIdsIdx = 0 at numFrame %d \n", numFrameRead);
            printf("FFT Result : \n");
            printf("fftRes[0] ");
            printComplex(fftRes[0]);
            for(int i = 0; i < 5; i ++ ){
                printComplex(fftRes[i]);
            }
            printf("Input frameRxData : \n");
            for(int i = 0; i < 5; i ++ ){
                printComplex(frameRxData[i]);
            }

            printf("Input baseFrameRx0 Data : \n");
            for(int i = 0; i < 5; i ++ ){
                printComplex(baseFrameRx0[i]);
            }
        }

        double maxDis = c * (((double) maxDisIdx / extendSize) * Fs_extend) / (2 * mu);
        distance[numFrameRead] = maxDis;
        printf("maxDisIdx %d numFrame %d \n", maxDisIdx, numFrameRead);


    }
//    for (int i = 0; i < FrameSize; i++) {
//        printf("distance[%d] %.5f \n", i, distance[i]);
//    }

    free(fftRes);
    free(frameRxData);
    free(frameDataReshaped);
    free(frameDataRx0);

//    free(distance);
    free(baseFrameReshaped);
    free(baseFrameRx0);
    free(inputData);
    fclose(fp);

}

int main() {
    test();
//    char filepath[] = "./fhy_direct.bin";
//    double *Dis = (double *) malloc((FrameSize) * sizeof(double));
//    DetectDis(filepath, Dis);
//    free(Dis);

    return 0;
}

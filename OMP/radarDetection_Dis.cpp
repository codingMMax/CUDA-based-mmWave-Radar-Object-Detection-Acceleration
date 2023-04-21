#include <stdio.h>
#include <math.h>
// #include <iostream>
// #include <cmath>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <ctime>
#include "FFT_OMP.h"

// using namespace std;

#define SampleSize 100              // the sample number in a chirp, suggesting it should be the power of 2
#define ChripSize 128               // the chirp number in a frame, suggesting it should be the the power of 2
#define FrameSize 1                 // the frame number
#define RxSize 4                    // the rx size, which is usually 4
#define cspd 3.0e8                     // the speed of light 
#define pi 3.141592653589793               // pi 



double F0=77e9;                      // the initial frequency
double mu=5.987e12;                  // FM slope
double chirp_sample=100;             // the sample number in a chirp, suggesting it should be the power of 2
double Fs=2.0e6;                     // sampling frequency
double num_chirp=128;                // the chirp number in a frame, suggesting it should be the the power of 2
double Framenum=90;                  // the frame number 
double Tr=64e-6;                     // the interval of the chirp
double fr=1/ Tr;                     // chirp repeating frequency,
double lamda=cspd/F0;                   // lamda of the initial frequency 
double d=0.5*lamda;                  // rx_wire array distance. When it is equal to the half of the wavelength, the 
                                     // maximum unambiguous Angle can reach -90° to +90°
int Tx_num = 1;
int Rx_num = 4;                      // the rx size, which is usually 4
 

// complex struct and complex algorithm
struct Complex{
    double real, imag;
};

Complex GetComplex(double r, double i){
    Complex temp;
    temp.real = r;
    temp.imag = i;
    return temp;
}

Complex Complex_ADD(Complex a, Complex b){
    Complex temp;
    temp.real = a.real + b.real;
    temp.imag = a.imag + b.imag;
    return temp;
}

Complex Complex_SUB(Complex a, Complex b){
    Complex temp;
    temp.real = a.real - b.real;
    temp.imag = a.imag - b.imag;
    return temp;
}

Complex Complex_MUL(Complex a, Complex b){
    Complex temp;
    temp.real = a.real * b.real - a.imag * b.imag;
    temp.imag = a.real * b.imag + a.imag * b.real;
    return temp;
}

Complex Complex_CDiv(Complex a, double b){
    Complex temp;
    temp.real = a.real /b;
    temp.imag = a.imag /b;
    return temp;
}

void Complex_MetricMUL(Complex *M_res, Complex *M_A, Complex *M_B, int sizea, int sizeb, int sizec){   
    // M_A = a*b
    // M_B = b*c
    // M_res = a*c
    for(int i =0;  i< sizea; i++){
        for(int j =0;  j< sizeb; j++){
            for(int k =0;  k< sizec; k++){
                *(M_res + i*sizea + k)=Complex_ADD(Complex_MUL((*(M_A + i*sizea + j)),(*(M_B + j*sizeb + k))),
                                        *(M_res + i*sizea + k));
            }
        }
    }
}

double Complex_mol(Complex *a){
    return sqrt(a->real*a->real+a->imag*a->imag);
}

// FFT recursive version
// Input : a pointer to a Complex Array, the size of the array
// Output: rewrite the output to x[]
void FFT_recursive(Complex x[], int len){
    Complex* odd, * even;
    Complex t;
    if (len == 1) return;
    odd = (Complex*)malloc(sizeof(Complex) * len / 2);
    even = (Complex*)malloc(sizeof(Complex) * len / 2);

    for (int i = 0; i < len / 2; i++){
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    FFT_recursive(odd, len / 2);
    FFT_recursive(even, len / 2);

    for (int i = 0; i < len / 2; i++){
        t = Complex_MUL(GetComplex(cos(2.0 * pi * i / len), -sin(2.0 * pi * i / len)), odd[i]);
        x[i] = Complex_ADD(even[i], t);
        x[i + len / 2] = Complex_SUB(even[i], t);
    }

    free(odd);
    free(even);
}
 
// FFT nonrecursive version
// Input : a pointer to a Complex Array, the size of the array
// Output: rewrite the output to x[]
void FFT_nonrecursive(Complex x[], int len){
    int temp = 1, l = 0;
    int *r = (int *)malloc(sizeof(int)*len);
    Complex t;
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
    for (int mid = 1; mid < len; mid <<= 1){
        Complex Wn = GetComplex(cos(pi / mid), -sin(pi / mid)); /*drop the "-" sin，then divided by len to get the IFFT*/
        for (int R = mid << 1, j = 0; j < len; j += R){
            Complex w = GetComplex(1, 0);
            for (int k = 0; k < mid; k++, w = Complex_MUL(w, Wn)){
                Complex a = x[j + k], b = Complex_MUL(w, x[j + mid + k]);
                x[j + k] = Complex_ADD(a, b);
                x[j + mid + k] = Complex_SUB(a, b);
            }
        }
    }
    free(r);
}

// FFT nonrecursive version
// Input : a pointer to a Complex Array, the size of the array
// Output: rewrite the output to x[]
void FFT_nonrecursive_OMP(Complex x[], int len){
    int temp = 1, l = 0;
    int *r = (int *)malloc(sizeof(int)*len);
    Complex t;
    //FFT index reverse，get the new index，l=log2(len)
    // #pragma omp parallel
    while (temp < len) temp <<= 1, l++;
    // #pragma omp parallel
    for (int i = 0; i < len; i++) r[i] = (r[i >> 1] >> 1) | ((i & 1) << (l - 1));
    //swap according to the index
    #pragma omp for schedule(static, 32) nowait
    for (int i = 0; i < len; i++)
        if (i < r[i]) {
            t = x[i];
            x[i] = x[r[i]];
            x[r[i]] = t;
        }
    // #pragma omp parallel
    // #pragma omp for
     
    int imax = (int)(log(len)/log(2));
    int R;
    // int mido;
    // printf("imax = %d\n",imax);
    //**********************************************Pragma***********************************************//
    // #pragma omp parallel // 
    //   {
    // #pragma omp single // pritvate(pnum)
    // {
    // #pragma omp task shared() untied if (nParticles > threshold && i<3)
    // #pragma omp task shared(x, R) untied final (R>256 )
    //**********************************************Pragma***********************************************//
    for (int i = 1; i < imax; i++){
    // for (int mid = 1; mid < len; mid = mid<< 1){
        int mid  = 1<< i ;
        // mido = mid;
        Complex Wn = GetComplex(cos(pi / mid), -sin(pi / mid)); /*drop the "-" sin，then divided by len to get the IFFT*/
        // printf("imax = %d, len = %d,mid = %d\n",imax, len,mido);
        R = mid << 1;
        
        //**********************************************Pragma***********************************************//
        // #pragma omp parallel
        // #pragma omp for
        // #pragma omp for schedule(static, 64) nowait
        #pragma omp task shared(R,x) untied final (R> 528)
        //**********************************************Pragma***********************************************//
       
        for (int j = 0; j < len; j += R){
            // printf("   j = %d, r= %d\n",j,R);
            Complex w = GetComplex(1, 0);
            // #pragma omp parallel
    
            for (int k = 0; k < mid; k++ ){
                w = Complex_MUL(w, Wn);
                Complex a = x[j + k];
                Complex b = Complex_MUL(w, x[j + mid + k]);
                x[j + k] = Complex_ADD(a, b);
                x[j + mid + k] = Complex_SUB(a, b);
            }
        }
    }
        //   }
        //}
      //}
    
    free(r);
}


// get the extended size
int getextendsize(int size){
    int pow2, powres;
    powres = 1;
    for (pow2 = 1; powres< size; pow2++ ){
        powres *= 2 ;
    } 
    return powres;
}

// FFTextend: extend and execute the FFT(recursive or nonrecursive)
// Input : *Data: a pointer to a Complex Array, size: the size of the array
// Output: *Res: FFT output of the extended input
void FFTextend(Complex *Data, Complex *Res,  int size){
    int powres, i;
    powres = getextendsize(size);
    
    Complex *Data_extend = (Complex *)malloc(powres*sizeof(Complex));
    memmove(Data_extend, Data, size *sizeof(Complex));
    for(i = size; i< powres; i++){
        (Data_extend+i)->real = 0;
        (Data_extend+i)->imag = 0;
    }
    
    FFT_nonrecursive(Data_extend, powres);
    // FFT_recursive(Data_extend, powres);

    memmove(Data, Data_extend, size *sizeof(Complex));
    memmove(Res, Data_extend, powres *sizeof(Complex));
    free(Data_extend);
}


void ccopy ( int n, double x[], double y[] )
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    y[i*2+0] = x[i*2+0];
    y[i*2+1] = x[i*2+1];
   }
  return;
}

void step ( int n, int mj, double a[], double b[], double c[],
  double d[], double w[], double sgn )
{
  double ambr;
  double ambu;
  int j;
  int ja;
  int jb;
  int jc;
  int jd;
  int jw;
  int k;
  int lj;
  int mj2;
  double wjw[2];

  mj2 = 2 * mj;
  lj = n / mj2;

 # pragma omp parallel \
    shared ( a, b, c, d, lj, mj, mj2, sgn, w ) \
    private ( ambr, ambu, j, ja, jb, jc, jd, jw, k, wjw )

 # pragma omp for nowait

  for ( j = 0; j < lj; j++ )
  {
    jw = j * mj;
    ja  = jw;
    jb  = ja;
    jc  = j * mj2;
    jd  = jc;

    wjw[0] = w[jw*2+0]; 
    wjw[1] = w[jw*2+1];

    if ( sgn < 0.0 ) 
    {
      wjw[1] = - wjw[1];
    }

    for ( k = 0; k < mj; k++ )
    {
      c[(jc+k)*2+0] = a[(ja+k)*2+0] + b[(jb+k)*2+0];
      c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];

      ambr = a[(ja+k)*2+0] - b[(jb+k)*2+0];
      ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1];

      d[(jd+k)*2+0] = wjw[0] * ambr - wjw[1] * ambu;
      d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;
    }
  }
  return;
}


void cfft2 ( int n, double x[], double y[], double w[], double sgn )
{
  int j;
  int m;
  int mj;
  int tgle;



   m = ( int ) ( log ( ( double ) n ) / log ( 1.99 ) );
   mj   = 1;
//
//  Toggling switch for work array.
//
  tgle = 1;
  step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );

  if ( n == 2 )
  {
    return;
  }
  
  for ( j = 0; j < m - 2; j++ )
  {
    mj = mj * 2;
    if ( tgle )
     {
       
      step ( n, mj, &y[0*2+0], &y[(n/2)*2+0], &x[0*2+0], &x[mj*2+0], w, sgn );
  
      tgle = 0;
    }
    else
    {
       
      step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
  
      tgle = 1;

    }
  }
 
//
//  Last pass thru data: move y to x if needed 
//
  
  if ( tgle ) 
  {
    ccopy ( n, y, x );
  }
  
  mj = n / 2;
 
  step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
 
  return;
}
//****************************************************************************80

void cffti ( int n, double w[] )
{
  double arg;
  double aw;
  int i;
  int n2;
  // const double pi = 3.141592653589793;

  n2 = n / 2;
  aw = 2.0 * pi / ( ( double ) n );

# pragma omp parallel \
    shared ( aw, n, w ) \
    private ( arg, i )

# pragma omp for nowait

  for ( i = 0; i < n2; i++ )
  {
    arg = aw * ( ( double ) i );
    w[i*2+0] = cos ( arg );
    w[i*2+1] = sin ( arg );
  }
  return;
}



void FFTextend_OMP(Complex *Data, Complex *Res,  int size){
    int powres, i;
    // time_t start, end, itime;    

    powres = getextendsize(size);
    //Complex *Data_extend = (Complex *)malloc(powres*sizeof(Complex));
    double *Data_double = (double *)malloc(2*powres*sizeof(double));
    double *y = (double *)malloc(2*powres*sizeof(double));
    double *w = (double *)malloc(powres*sizeof(double));
    // smemmove(Data_extend, Data, size *sizeof(Complex));
  
     for(i = 0; i< size; i++){
        *(Data_double+2*i) = (Data+i)->real;
        *(Data_double+2*i+1) = (Data+i)->imag;
    }
    for(i = size; i< powres; i++){
        *(Data_double+2*i) = 0;
        *(Data_double+2*i+1)  = 0;
    }

    // start =clock();
    cffti(powres, w);
    // itime =clock();
    cfft2(powres, Data_double, y, w, -1);
    // cfft2(powres, y,Data_double, w, -1);
    // end =clock();
    // printf("ffti time=%lfms,  fft2 time=%lfms  ", difftime(itime, start)/CLOCKS_PER_SEC*1000,difftime(end, itime)/CLOCKS_PER_SEC*1000);
    for(i = 0; i< powres; i++){
        (Res+i)->real= *(y+2*i+0) ;
        (Res+i)->imag= *(y+2*i+1) ;
    }
    // memmove(Data, Res, size *sizeof(Complex));
    // memmove(Res, Data_extend, powres *sizeof(Complex));
    // free(Data_extend);
    free(y);
    free(w);
    free(Data_double);
}

/// C read the binfile size
int getBinSize(char *path){
    int  size = 0;
    FILE  *fp = fopen(path, "rb");
    if (fp){
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fclose(fp);
    }
    return size;
}

// C read the bin data in size of short
void readBin(char *path, short *buf, int size){
    FILE *infile;
    if ((infile = fopen(path, "rb")) == NULL){
        printf("\nCan not open the path: %s \n", path);
    }
    fread(buf, sizeof(short), size, infile);
    fclose(infile);
}

// C write the bin data in size of short
void writeBin(char *path, short *buf, int size){
    FILE *outfile;
    if ((outfile = fopen(path, "wb")) == NULL){
        printf("\nCan not open the path: %s \n", path);
    }
    fwrite(buf, sizeof(short), size, outfile);
    fclose(outfile);
}

// C write the bin data in size of Complex
void writeComplexBin(Complex *buf, int size){
    char saveFilePath_real[] = "./res_reshape_real.bin"  ;
    char saveFilePath_imag[] = "./res_reshape_imag.bin" ;

    short *real = (short*)malloc(size*sizeof(short));
    short *imag = (short*)malloc(size*sizeof(short));

    Complex *ptr= buf;
    printf("\nwrite size = %d\n ",size);
    for (int i=0; i< size; i++){
        *(real+i) =(short)(ptr->real);
        *(imag+i) =(short)(ptr->imag);
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
void ReshapeComplex(short *OriginalArray, Complex *Reshape, int size){
    int i, j, k, l;
    int cnt = 0;
    Complex *buf_complex = (Complex *)malloc(size*sizeof(Complex)/2);
    short *ptr;
    Complex *complex_ptr = buf_complex;
    // reshape into 2 form of complex

    // #

    // # pragma omp for nowait
    // #pragma omp for schedule(static, 64) nowait
    for(i= 0 ;i < size; i+=4){
        ptr = OriginalArray+ i;
        complex_ptr->real = (double)*(ptr);  
        complex_ptr->imag  = (double)*(ptr +2 ); 
        complex_ptr++;  
        complex_ptr->real = (double)*(ptr +1 );  
        complex_ptr->imag  = (double)*(ptr +3 );   
        complex_ptr++;    
    }      
    Complex *Reshape_ptr;
    // change the sequence of the array, reshape it in form of Rx instead of frame
    
    // # pragma omp for nowait
    #pragma omp for schedule(static, 64) nowait
    for (i = 0; i <RxSize; i++){
        for (j = 0; j <FrameSize*ChripSize; j++){
            for(k =0; k< SampleSize; k++){
                Reshape_ptr = (Reshape+i*FrameSize*ChripSize*SampleSize+j*SampleSize+k); 
                complex_ptr = (buf_complex+j*RxSize*SampleSize+i*SampleSize+k);
                Reshape_ptr->real = complex_ptr->real;
                Reshape_ptr->imag = complex_ptr->imag;
            }
        }
    }
    free(buf_complex);
    return;
}

// read the file and call the "ReshapeComplex" to reshape
int readandreshape(char *filepath, Complex *Data_reshape){
    // ----------------------read size------------------------------
    char filePath[] = "./fhy_direct.bin";
    int bytesize = getBinSize(filePath);
    int size = bytesize / sizeof(short);
    // ----------------------read int16 ------------------------------
    short *buf = (short*)malloc(size*sizeof(short));
    readBin(filePath, buf, size);
    // ----------------------reshape ------------------------------
    short *buf_ptr = (short*)buf;
    short *buf_reshape_real = (short*)malloc(size*sizeof(short)/2);
    short *buf_reshape_imag  = (short*)malloc(size*sizeof(short)/2);
    Complex *buf_reshape = (Complex *)malloc(size*sizeof(Complex)/2);
    ReshapeComplex(buf_ptr, buf_reshape, size);
    return size/2;
}

// find the max 
int FindMax(double *ptr, int size){
    int maxidx = 0;
    int maxval = 0;
    for (int i = 0; i < size; i++){
        if(*(ptr) > maxval){
            maxval = *(ptr);
            maxidx = i;
        }
        ptr++;
    }
    return maxidx;
}

// find the max of the abs of the complex array
int FindAbsMax(Complex *ptr, int size){
    int maxidx = 0;
    double maxval = 0;
    double absval;
    for (int i = 0; i < size; i++){
        absval = Complex_mol((ptr+i));
        if(absval > maxval){
            maxval = absval;
            maxidx = i;
        }
    }
    return maxidx;
}


void printComplex(Complex *Data, int size){
    printf("Print!\n");
    for (int i = 0; i < size; i++){
      printf("data %d = %lf+i*%lf\n", i, (Data+i)->real,  (Data+i)->imag);
    }
}

void printAbs(Complex *Data, int size){
    printf("Print!\n");
    double real, imag, abs;
    for (int i = 0; i < size; i++){
        real = (Data+i)->real;
        imag = (Data+i)->imag;
        abs = sqrt(real*real +imag*imag);
        printf("dataabs %d = %lf\n", i, abs);
    }
}

// Detect the distance , 
// Input : the bin size file 
// Output: the dis array
int DectectDis(char *filepath, double *Dis){
    // ----------------------read int16 ------------------------------
    FILE *infile;
    if ((infile = fopen(filepath, "rb")) == NULL){
        printf("\nCan not open the path: %s \n", filepath);
    }
    // FILE *infile2;
    // if ((infile = fopen(filepath, "rb")) == NULL){
    //     printf("\nCan not open the path: %s \n", filepath);
    // }
    
    
    int readsize = ChripSize*SampleSize*RxSize*2 ; 
    int size = ChripSize*SampleSize*RxSize;
    
    short *Data_Frm0 = (short*)malloc( readsize * sizeof(short));
    //readBin(filepath, Data_Frm0, readsize);
    fread(Data_Frm0, sizeof(short), readsize, infile);
    Complex *Data_Frm0_reshape = (Complex *)malloc( size * sizeof(Complex));
    ReshapeComplex(Data_Frm0, Data_Frm0_reshape, readsize);
    // printComplex(Data_Frm0_reshape, 100);

    // malloc for the data
    short *Data_Frm = (short*)malloc(readsize * sizeof(short));
    Complex *Data_Frm_reshape = (Complex *)malloc( size *sizeof(Complex));
    // Complex *Data_Frm_reshape_rx0 = (Complex *)malloc( ChripSize * SampleSize * sizeof(Complex));
    Complex *Data_Frm_rx = (Complex *)malloc( ChripSize * SampleSize * sizeof(Complex));
    Complex *Data_fft = (Complex *)malloc(getextendsize(ChripSize*SampleSize)* sizeof(Complex));


    // caculate the position and speed
    int frm, chp, smp;
    int testfrm = 1;
    time_t start, end, fftstart, fftend, rpstart, rpend, readstart, readend, maxstart, maxend;
    double cost, fftcost, reshapecost, readcost;  
    double totaltime , ffttime, rptime, readtime, maxtime;
    int cnt;
    totaltime = 0; ffttime = 0; rptime = 0; readtime = 0; maxtime =0, cnt = 0;
    frm = 0;
    // for (frm = testfrm; frm < testfrm+60; frm++ ){
    while(1){
        //if(frm ==90 )break;

        start= clock();
        readstart = clock();
        // ----------------------- read data ------------------------
        if(fread(Data_Frm, sizeof(short), ChripSize*SampleSize*RxSize*2, infile) == 0){
            break;
        }
        // ----------------------- read data ------------------------
        readend= clock();

        // if (fread(Data_Frm, sizeof(short), ChripSize*SampleSize*RxSize*2, infile) == 0){
        //     printf("\nCan not read the path: %s \n", filepath);
        // }

        frm++;
        

        rpstart = clock();
        // ----------------------- reshape data ------------------------
        ReshapeComplex(Data_Frm, Data_Frm_reshape, readsize);
        
        // memmove(Data_Frm_reshape_rx0, Data_Frm_reshape, ChripSize*SampleSize*sizeof(Complex));
        // ----------------------- resahpe data ------------------------
       

        for (int i = 0; i < ChripSize * SampleSize; i++){
            *(Data_Frm_rx+i) = Complex_SUB(*(Data_Frm_reshape+i), *(Data_Frm0_reshape+i));
        
        }
        rpend = clock();
        //---------------------------------------------------distance----------------------------------------------------
        fftstart = clock();
        // ----------------------- FFT data ------------------------
        Complex *frm_ptr = Data_Frm_rx;
        // FFTextend(Data_Frm_rx, Data_fft, ChripSize*SampleSize);
        FFTextend_OMP(Data_Frm_rx, Data_fft, ChripSize*SampleSize);
        // ----------------------- FFT data ------------------------
        fftend = clock();
        // printComplex( Data_fft, 100);

        maxstart = clock();
        // ----------------------- MAX data ------------------------
        int extendsize = getextendsize(ChripSize*SampleSize);
        double Fs_extend = Fs * extendsize/(ChripSize*SampleSize);
        int maxDisidx = FindAbsMax(Data_fft, floor(0.4*extendsize))*(ChripSize*SampleSize)/extendsize;
        double maxDis = cspd*(((double)maxDisidx/extendsize)*Fs_extend)/(2*mu);
        *(Dis+frm) = maxDis ;
        // ----------------------- MAX data ------------------------
        maxend = clock();

        end = clock();
        
        totaltime += difftime(end,start)/CLOCKS_PER_SEC*1000;
        ffttime +=difftime(fftend,fftstart)/CLOCKS_PER_SEC*1000;
        readtime += difftime(readend,readstart)/CLOCKS_PER_SEC*1000;
        rptime+=difftime(rpend,rpstart)/CLOCKS_PER_SEC*1000;
        maxtime +=difftime(maxend,maxstart)/CLOCKS_PER_SEC*1000;
        cnt+=1;

        // cost=difftime(fftend,fftstart)/CLOCKS_PER_SEC*1000; 
        // if(cost <1 && difftime(readend,readstart)/CLOCKS_PER_SEC*1000 <1 && difftime(rpend,rpstart)/CLOCKS_PER_SEC*1000 <1  && difftime(maxend,maxstart)/CLOCKS_PER_SEC*1000 <1 ) {
        //    totaltime += difftime(end,start)/CLOCKS_PER_SEC*1000;
        //     ffttime +=difftime(fftend,fftstart)/CLOCKS_PER_SEC*1000;
        //     readtime += difftime(readend,readstart)/CLOCKS_PER_SEC*1000;
        //     rptime+=difftime(rpend,rpstart)/CLOCKS_PER_SEC*1000;
        //     maxtime +=difftime(maxend,maxstart)/CLOCKS_PER_SEC*1000;
        //     cnt+=1;
        // } 

 
    }
    
    fclose(infile);
    free(Data_Frm);
    free(Data_Frm_reshape);
    // free(Data_Frm_reshape_rx0);
    free(Data_Frm_rx);
    free(Data_fft);
    free(Data_Frm0);
    free(Data_Frm0_reshape);

    printf("\ncnt=%d, totaltime=%fms, thp=%ffrm/s read=%fms,  reshape=%fms,  fft=%fms,  findmax=%fms\n",
    cnt ,  totaltime, 1000*89/totaltime,
    readtime,  rptime,  
    ffttime , maxtime); 
    return frm;
    
}


   

int main(){
    char filepath[] = "./fhy_direct.bin";
    double *Dis = (double *)malloc(90 * sizeof(double));
    time_t totalstart, totalend;
    // time(&totalstart);
    totalstart = clock();
    // testOMP();
    int frm;
    frm= DectectDis(filepath, Dis);
    time(&totalend);
    totalend = clock();
    double cost=difftime(totalend,totalstart); 
    printf("total throughput = %lf frm/s \n",1/(cost/frm/CLOCKS_PER_SEC));
    free(Dis);
}
#ifndef ACCELERATION_H
#define ACCELERATION_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


// complex struct and complex algorithm
struct Complex_t {
    double real, imag;
};

double cudaProcessing(short* deviceIn, Complex_t* host_baseFrame, int size);

#endif
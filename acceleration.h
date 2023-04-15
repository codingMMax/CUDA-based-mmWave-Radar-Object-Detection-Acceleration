#ifndef ACCELERATION_H
#define ACCELERATION_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

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

// complex struct and complex algorithm
struct Complex_t
{
  double real, imag;
};

double cudaProcessing(short *deviceIn, Complex_t *host_baseFrame, int size, double *fftTime, double *preProcessTime, double *findMaxTime, double *totalTime);

#endif
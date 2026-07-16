#ifndef FFT_H
#define FFT_H

#include <vector>
#include <complex>

void fft_radix2(const std::vector<std::complex<float>>& input,
                std::vector<std::complex<float>>& output);

void fft_radix2_openmp(const std::vector<std::complex<float>>& input,
                       std::vector<std::complex<float>>& output);

#endif

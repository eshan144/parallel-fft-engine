#ifndef DFT_H
#define DFT_H

#include <vector>
#include <complex>

void dft_naive(const std::vector<std::complex<float>>& input,
               std::vector<std::complex<float>>& output);

#endif

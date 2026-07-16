#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}

    double elapsed_ms() const {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0;
    }

    void reset() {
        start_ = std::chrono::steady_clock::now();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

#endif

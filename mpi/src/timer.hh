#pragma once
#include <chrono>

namespace icesp
{

struct timer
{
    // and other clock e.g. system_clock, steady_clock
    using value_type      = double;
    using clock_type      = std::chrono::high_resolution_clock;
    using time_point_type = std::chrono::time_point<clock_type>;
    using duration_type   = std::chrono::duration<value_type>;

    void restart()
    {
        reset();
        start();
    }

    void start()
    {
        if (started) return;
        _start = clock_type::now();
        started = true;
    }

    void stop()
    {
        using namespace std::chrono;
        end = clock_type::now();
        auto elapsed = duration_cast<milliseconds>(end - _start).count() / 1000.;
        tot += elapsed;
        started = false;
    }

    void reset()
    {
        tot = 0;
        started = false;
    }

    auto elapsed_seconds() const
    {
        return tot;
    }

private:
    time_point_type _start;
    time_point_type end;
    value_type tot{};
    bool started{};
};

} // namespace icesp


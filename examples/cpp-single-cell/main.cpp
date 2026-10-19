// Solve a single cell of two different models in one C++ program, with the code generated
// by `gotranx ode2cpp`.
//
// Usage: main <output.csv>
//
// Each generated header puts its model in a namespace (tor_ord and ten_tusscher), so the
// two models' functions and constants (rhs, NUM_STATES, ...) do not clash.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ten_tusscher.hpp"
#include "tor_ord.hpp"

static int lookup(int index, const char *name)
{
    if (index < 0) {
        std::fprintf(stderr, "No state named '%s'\n", name);
        std::exit(1);
    }
    return index;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <output.csv>\n", argv[0]);
        return 1;
    }

    const double dt = 0.01;      // ms
    const double t_end = 1000.0; // ms, one beat
    const int save_every = 10;   // save every 0.1 ms

    // ToR-ORd
    std::vector<double> tor_ord_parameters(tor_ord::NUM_PARAMS);
    std::vector<double> tor_ord_states(tor_ord::NUM_STATES);
    std::vector<double> tor_ord_values(tor_ord::NUM_STATES);
    tor_ord::init_parameter_values(tor_ord_parameters.data());
    tor_ord::init_state_values(tor_ord_states.data());
    const int tor_ord_v = lookup(tor_ord::state_index("v"), "v");

    // ten Tusscher-Panfilov
    std::vector<double> ten_tusscher_parameters(ten_tusscher::NUM_PARAMS);
    std::vector<double> ten_tusscher_states(ten_tusscher::NUM_STATES);
    std::vector<double> ten_tusscher_values(ten_tusscher::NUM_STATES);
    ten_tusscher::init_parameter_values(ten_tusscher_parameters.data());
    ten_tusscher::init_state_values(ten_tusscher_states.data());
    const int ten_tusscher_v = lookup(ten_tusscher::state_index("V"), "V");

    std::FILE *out = std::fopen(argv[1], "w");
    if (out == nullptr) {
        std::fprintf(stderr, "Could not open %s\n", argv[1]);
        return 1;
    }
    std::fprintf(out, "t,v_tor_ord,v_ten_tusscher\n");

    const int num_steps = static_cast<int>(t_end / dt);
    for (int step = 0; step <= num_steps; step++) {
        const double t = step * dt;
        if (step % save_every == 0) {
            std::fprintf(out, "%g,%.10g,%.10g\n", t, tor_ord_states[tor_ord_v],
                         ten_tusscher_states[ten_tusscher_v]);
        }
        // Each scheme writes the states at t + dt into its `values`
        tor_ord::generalized_rush_larsen(tor_ord_states.data(), t, dt, tor_ord_parameters.data(),
                                         tor_ord_values.data());
        tor_ord_states.swap(tor_ord_values);
        ten_tusscher::generalized_rush_larsen(ten_tusscher_states.data(), t, dt,
                                              ten_tusscher_parameters.data(),
                                              ten_tusscher_values.data());
        ten_tusscher_states.swap(ten_tusscher_values);
    }

    std::fclose(out);
    return 0;
}

#include <astrolabio/astrolabio.hpp>
#include <vector>

using namespace astrolabio;
using CppAD::AD;

using real = double;

struct ParticleParams {
    real  mass;
};

template <typename T>
std::vector<T> ode(const ParticleParams* params,
                   const std::vector<T>& state, const std::vector<T>& alg, const std::vector<T>& inputs)
{
    return std::vector<T>{
        state[1],
        inputs[0] / params->mass,
    };
}

template <typename T>
std::vector<T> alg(const ParticleParams* params, const std::vector<T>& state, const std::vector<T>& alg, const std::vector<T>& inputs)
{
    return std::vector<T>{};
}

int main() {
    using Solver = IDASolver<ParticleParams, real>;
    int num_diffeq = 2;
    int num_algeq = 0;
    int num_algvar = 0;
    int num_inputs = 1;
    auto solver = Solver(num_diffeq, num_algeq, num_algvar, num_inputs, 
        &ode<real>,
        &alg<real>,
        &ode<AD<real>>,
        &alg<AD<real>>);

    ParticleParams p = {
        .mass = 10,
    };

    std::vector<real> avtol(num_diffeq + num_algvar, 1e-4);
    solver.set_avtol(avtol);
    solver.set_rtol(1e-5);

    std::vector<real> states_init(num_diffeq + num_algvar, 0.0);
    std::vector<real> input_init(num_inputs, 10.0);
    solver.set_params(&p);
    solver.set_initial_state(states_init, input_init);

    real t0 = 0;
    real step = 1e-3;
    real tf = 1000;
    solver.start(t0, step, false);

    std::cout << "Starting loop" << std::endl;
    solver.set_input({1});
    double tret;
    std::vector<real> x, z;
    while (tret < tf) {
        solver.get_state(x, z);
        int retval = solver.do_step(step, tret);

        if (retval == IDA_TSTOP_RETURN) {
            break;
        } else if (retval < 0) {
            fprintf(stderr, "IDA error while solving: %d\n", retval);
            break;
        }
    }
    std::cout << "Final position: " << x[0] << "m" << std::endl;
    std::cout << "Theorical position: " << (0.5*1.0/p.mass*tf*tf) << "m" << std::endl;
}

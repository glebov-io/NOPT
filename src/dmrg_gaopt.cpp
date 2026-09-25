// dmrg_gaopt — block2's genetic orbital-ordering driver, as pyblock2 runs it.

#include "dmrg_gaopt.h"

#include <algorithm>
#include <set>

#include "dmrg/orbital_ordering.hpp"

using namespace block2;

// pyblock2's driver settings. n_configs and n_elite differ from ga_opt's own defaults, so every
// parameter is passed explicitly. The elite count is capped by the population (n_sites < 4).
static const int    DMRG_GAOPT_GENERATIONS = 10000;
static const int    DMRG_GAOPT_ELITE       = 8;
static const double DMRG_GAOPT_CLONE_RATE  = 0.1;
static const double DMRG_GAOPT_MUTATE_RATE = 0.1;

std::vector<uint16_t> dmrg_gaopt_order(int n_sites, const std::vector<double> &kmat, int n_tasks) {
    // The set dedups and sorts the tasks' permutations; its first element is the one kept.
    std::set<std::vector<uint16_t>> orders;
    for (int i_task = 0; i_task < n_tasks; i_task++) {
        Random::rand_seed(1234 + i_task); // one process-wide RNG: the tasks must stay serial
        orders.insert(OrbitalOrdering::ga_opt((uint16_t)n_sites, kmat, DMRG_GAOPT_GENERATIONS,
                                              2 * n_sites, std::min(DMRG_GAOPT_ELITE, 2 * n_sites),
                                              DMRG_GAOPT_CLONE_RATE, DMRG_GAOPT_MUTATE_RATE));
    }
    return *orders.begin();
}

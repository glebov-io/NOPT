#pragma once
//
// dmrg_gaopt — genetic-algorithm orbital ordering, a port of pyblock2's gaopt driver.
// Deterministic: fixed seeds 1234+task, lexicographically smallest of the distinct results.
// kmat is the row-major n_sites x n_sites coupling matrix.

#include <cstdint>
#include <vector>

// pyblock2's task count. The tasks collapse to a handful of distinct orders whose costs spread by
// ~0.01%, so a reduced count is enough to price an order but not to pick the one a solve runs on.
constexpr int DMRG_GAOPT_TASKS = 64;

std::vector<uint16_t> dmrg_gaopt_order(int n_sites, const std::vector<double> &kmat,
                                       int n_tasks = DMRG_GAOPT_TASKS);

#ifndef __superci_pt
#define __superci_pt

# include <vector>

# include "orbital_lbfgs.h"

class CAS_engine;

// Perturbative super-CI orbital converger ($CAS converger=sxpt). The first-order amplitudes
// of the Dyall H0 are the bare orbital step and the inverse Hessian at the centre of an L-BFGS;
// two n_act x n_act pencils serve every core and every virtual orbital. State-averaged only --
// the Koopmans matrices are built from the SA 1- and 2-RDM.
class superci_pt_engine{
    public:
        int    init(int ext_n_c, int ext_n_a, int ext_n_v, int ext_n_ao,
                    const int * ext_rep_num, int ext_n_rep, bool ext_sym, double ext_x_max,
                    int ext_lbfgs);
        double calc(const double * G);      // max|g| over the three rotatable blocks
        double step(CAS_engine * CAS, double s_conv);   // build kappa from the L-BFGS direction; apply it unless max|T| < s_conv; returns max|T|
        double applied() const { return app_max; }   // max|rotation actually applied| by the last step()
        void   reset_history();             // drop everything carried across macro-iterations

    private:
        int n_c, n_a, n_v, n_ao, n_mo, n_rep;
        const int * rep_num;
        bool   sym;                          // blocks and pencils per irrep; false: one block, labels all -1
        double x_max;
        double app_max;
        bool   kept_changed;                 // a pencil's kept metric set changed this macro-iteration
        orbital_lbfgs lbfgs;

        // canonical frame: V[p*dim+mu] holds eigenvectors as columns, mu reusing the
        // window's own slots (no permutation), eps the matching orbital energies
        std::vector<double> V_c, V_v, eps_c, eps_v;
        std::vector<double> K, K_t;                  // Koopmans matrices, n_a x n_a
        struct sx_pencil{
            std::vector<double> C, eig;              // pencil solutions, rows are vectors
            std::vector<double> D;                   // dropped metric eigenvectors, nd rows
            std::vector<int>    rep;                 // irrep of each pencil row, -1 without symmetry
            std::vector<int>    keep;                // kept counts per irrep, previous macro-iter
            int                 nd;
            bool                reported;            // one drop NOTE per pencil, not per engine
        };
        sx_pencil pp, ph;                            // particle and hole pencils
        std::vector<double> gc, Tc;                  // amplitudes() scratch: its input and output in the canonical frame
        std::vector<double> kappa, buf1, buf2;
        std::vector<double> T_vec, g_vec, app_vec;

        void members(int base, int dim, int i_r, std::vector<int>& mem) const;
        void canonicalize_block(const double * F, int n0, int dim,
                                std::vector<double>& V, std::vector<double>& eps);
        double canonical_residual(const double * F, int n0, int dim,
                                  const std::vector<double>& V);
        double orthogonality_defect(int dim, const std::vector<double>& V);
        double min_denominator() const;             // min PT denominator over the symmetry-allowed pairs
        void build_koopmans(CAS_engine * CAS);
        void solve_pencil(const double * M_in, const double * metric, double sign,
                          sx_pencil & P, const char * what);
        void amplitudes(const double * q, std::vector<double>& T);   // T = -M^-1 q in the frame step() built
        void apply_rotation(CAS_engine * CAS);
};

#endif

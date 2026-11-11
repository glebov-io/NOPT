#ifndef NOPT_TENSOR_TAPE_H
#define NOPT_TENSOR_TAPE_H

// A small reverse-mode tape over dense tensors (analytic gradients: the DSRG-PT2 energy is transcribed onto it term by term).
// Every operation records its adjoint; backward() runs them in reverse. Operations:
//   einsum   out[o] (+)= alpha * prod_k in_k[i_k]   (index letters, summation over the letters absent from the output; a letter
//            repeated inside one operand takes a diagonal). Adjoint of each operand: the same product with that factor left out.
//   reg      out = x * f(D), D = sum_axis sign_axis * eps_axis[i_axis]  (the DSRG regularizers, elementwise; adjoints to x and to the eps).
//   auxgemm  out[a1,a2,b1,b2] = sum_P A[a1,a2,P] B[b1,b2,P]  (RI three-index -> four-index, by GEMM).
//   lin      out = sum_k c_k X_k  (same shape).
// The loops are plain nested loops (no BLAS except auxgemm): cost per einsum = product of the extents of all its letters.
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tt{

struct Ten{
    std::vector<int> dim;
    std::vector<double> v, g;
    bool grad = true;            // false: a constant (no adjoint accumulated)
    size_t size() const { size_t n = 1; for(int d : dim) n *= (size_t)d; return n; }
};
using P = std::shared_ptr<Ten>;

enum class Reg{ Q, ONE_PLUS_Q, R, QR, INV };   // q = exp(-s D^2), 1 + q, R_s(D) = (1 - q)/D (Taylor branch as in production), (1+q) R_s, 1/D

class Tape{
public:
    P leaf(const std::vector<int> & dim, const double * data, bool grad = true);
    P zeros(const std::vector<int> & dim, bool grad = true);
    P einsum(const std::string & spec, const std::vector<P> & ins, double alpha = 1.0);
    void einsum_into(const P & out, const std::string & spec, const std::vector<P> & ins, double alpha = 1.0);
    P lin(const std::vector<std::pair<double, P>> & terms);
    // eps[k] == nullptr: axis k carries no orbital energy
    P reg(const P & x, Reg kind, double s, const std::vector<P> & eps, const std::vector<double> & sign);
    P auxgemm(const P & A, const P & B);
    void backward(const P & scalar, double seed = 1.0);
    size_t n_ops() const { return back_.size(); }
private:
    std::vector<std::function<void()>> back_;
};

double reg_value(Reg kind, double s, double D);
double reg_deriv(Reg kind, double s, double D);

}   // namespace tt

#endif

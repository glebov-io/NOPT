# include "cdas_fold.h"
# include "blas_link.h"
# include <omp.h>
# include <vector>

namespace {

// Pins BLAS to one thread and OpenMP to num_threads for the lifetime of the object: the OpenMP
// loops below own the cores and call BLAS from inside the parallel region. Restores both on exit,
// OpenMP last: an OpenMP-threaded OpenBLAS up to 0.3.21 resets it in openblas_set_num_threads.
struct blas_serial_guard {
    int saved;
    int omp_saved;
    blas_serial_guard() : saved(1), omp_saved(omp_get_max_threads()) {
#ifdef _OPENBLAS
        saved = openblas_get_num_threads();
        openblas_set_num_threads(1);
#endif
#ifdef _MKL
        saved = mkl_get_max_threads();
        mkl_set_num_threads(1);
#endif
        omp_set_num_threads(num_threads);
    }
    ~blas_serial_guard() {
#ifdef _OPENBLAS
        openblas_set_num_threads(saved);
#endif
#ifdef _MKL
        mkl_set_num_threads(saved);
#endif
        omp_set_num_threads(omp_saved);
    }
    blas_serial_guard(const blas_serial_guard &) = delete;
    blas_serial_guard &operator=(const blas_serial_guard &) = delete;
};

}

void assemble_g3(int n, const double * T3_AB, double * g3){
    auto ix = [n](int t,int u,int v,int w,int x,int y)->size_t {
        return (((((size_t)t*n+u)*n+v)*n+w)*n+x)*n+y; };

    #pragma omp parallel for collapse(2) schedule(static)
    for(int t=0;t<n;t++)
    for(int u=0;u<n;u++)
    for(int v=0;v<n;v++)
    for(int w=0;w<n;w++)
    for(int x=0;x<n;x++)
    for(int y=0;y<n;y++){
       g3[ix(t,u,v,w,x,y)] =
              ( 2.0*T3_AB[ix(t,v,u,w,x,y)]
              + 2.0*T3_AB[ix(t,x,u,y,v,w)]
              + 2.0*T3_AB[ix(v,t,w,u,x,y)]
              + 2.0*T3_AB[ix(v,x,w,y,t,u)]
              + 2.0*T3_AB[ix(x,t,y,u,v,w)]
              + 2.0*T3_AB[ix(x,v,y,w,t,u)]
              -     T3_AB[ix(t,v,u,w,x,y)]
              +     T3_AB[ix(v,t,u,w,x,y)]
              +     T3_AB[ix(x,v,u,w,t,y)]
              +     T3_AB[ix(t,x,u,w,v,y)]
              -     T3_AB[ix(v,x,u,w,t,y)]
              -     T3_AB[ix(x,t,u,w,v,y)] ) / 12.0;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for(int t=0;t<n;t++) for(int u=0;u<n;u++)
    for(int v=0;v<n;v++) for(int w=0;w<n;w++)
    for(int x=0;x<n;x++) for(int y=0;y<n;y++){
        const size_t i = ix(t,u,v,w,x,y), id = ix(u,t,w,v,y,x);
        if(i < id){ double s = 0.5*(g3[i]+g3[id]); g3[i] = g3[id] = s; }
    }
}

void build_fold(int n, const double * g3, const double * gamma, const double * GAMMA,
                double * F1, double * F2){
    if(n <= 0) return;
    const size_t n2 = (size_t)n*n, n3 = n2*(size_t)n, n4 = n2*n2;
    const int in = n, in2 = (int)n2, in4 = (int)n4;

    // the cross contraction C[vw,xy] = sum_tu gamma_tu g3[tw,vu,xy]; not pair symmetric alone
    std::vector<double> C(n4, 0.0);

    blas_serial_guard bsg;

    // D[vw,xy] = sum_tu gamma_tu g3[tu,vw,xy] on the (n^2)x(n^4) reshape of g3, and C by its
    // trailing (x,y) block; one output pair block (v,w) per iteration, so no writer collides.
    #pragma omp parallel for collapse(2) schedule(static)
    for(int v=0;v<n;v++)
    for(int w=0;w<n;w++){
        const size_t off = ((size_t)v*n+w)*n2;
        cblas_dgemv(CblasRowMajor,CblasTrans,
                    in2,in2,1.0,
                    g3+off, in4,
                    gamma,1,
                    0.0, F2+off,1);
        for(int t=0;t<n;t++)
            cblas_dgemv(CblasRowMajor,CblasTrans,
                        in,in2,1.0,
                        g3+(((size_t)t*n+w)*n+v)*n3, in2,
                        gamma+(size_t)t*n,1,
                        (t==0 ? 0.0 : 1.0), C.data()+off,1);
    }

    // F2 = D - 1/2 (C + C^T), the pair-symmetric representative
    #pragma omp parallel for collapse(2) schedule(static)
    for(int v=0;v<n;v++)
    for(int w=0;w<n;w++)
    for(int x=0;x<n;x++)
    for(int y=0;y<n;y++){
        const size_t i = (((size_t)v*n+w)*n+x)*n+y;
        const size_t j = (((size_t)x*n+y)*n+v)*n+w;
        F2[i] -= 0.5*(C[i]+C[j]);
    }

    // f1'[a,b] = 1/2 sum_vwxy g3[ab,vw,xy] GAMMA[vw,xy] - 1/2 sum_uvxy g3[au,vb,xy] GAMMA[vu,xy];
    // the cross term contracts the trailing (x,y) block and accumulates over the free b.
    #pragma omp parallel for schedule(static)
    for(int a=0;a<n;a++){
        std::vector<double> acc(n, 0.0);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,
                    in,in4,0.5,
                    g3+(size_t)a*n*n4, in4,
                    GAMMA,1,
                    0.0, F1+(size_t)a*n,1);
        for(int u=0;u<n;u++)
        for(int v=0;v<n;v++)
            cblas_dgemv(CblasRowMajor,CblasNoTrans,
                        in,in2,1.0,
                        g3+(((size_t)a*n+u)*n+v)*n3, in2,
                        GAMMA+((size_t)v*n+u)*n2,1,
                        1.0, acc.data(),1);
        for(int b=0;b<n;b++) F1[(size_t)a*n+b] -= 0.5*acc[b];
    }

    // spill of {e_{vw,xy}} -> e_{vw,xy}: F1 = f1' - sum_xy F2[ab,xy] gamma_xy
    //                                        + 1/2 sum_xy F2[ay,xb] gamma_xy
    #pragma omp parallel for collapse(2) schedule(static)
    for(int a=0;a<n;a++)
    for(int b=0;b<n;b++){
        const double d = cblas_ddot(in2, F2+(((size_t)a*n+b)*n2),1, gamma,1);
        double c = 0.0;
        for(int x=0;x<n;x++)
        for(int y=0;y<n;y++)
            c += F2[(((size_t)a*n+y)*n+x)*n+b] * gamma[(size_t)x*n+y];
        F1[(size_t)a*n+b] += 0.5*c - d;
    }
}

double fold_e1(int n, const double * h1, const double * gamma){
    if(n <= 0) return 0.0;
    return cblas_ddot((int)((size_t)n*n), h1,1, gamma,1);
}

double fold_e2(int n, const double * h2, const double * GAMMA){
    if(n <= 0) return 0.0;
    const size_t n2 = (size_t)n*n;
    return 0.5*cblas_ddot((int)(n2*n2), h2,1, GAMMA,1);
}

double fold_e3(int n, const double * g3, const double * G3){
    if(n <= 0) return 0.0;
    double e = 0.0;
    blas_serial_guard bsg;
    // <e_{tu,vw,xy}> = G3[t,v,x,u,w,y]; both sides are contiguous in the trailing y
    #pragma omp parallel for collapse(2) schedule(static) reduction(+:e)
    for(int t=0;t<n;t++)
    for(int u=0;u<n;u++)
    for(int v=0;v<n;v++)
    for(int w=0;w<n;w++)
    for(int x=0;x<n;x++)
        e += cblas_ddot(n, g3+(((((size_t)t*n+u)*n+v)*n+w)*n+x)*n,1,
                           G3+(((((size_t)t*n+v)*n+x)*n+u)*n+w)*n,1);
    return e/6.0;
}

double fold_scalar(int n, const double * g3, const double * G3_ens, const double * F1,
                   const double * F2, const double * gamma, const double * GAMMA){
    return fold_e3(n, g3, G3_ens) - fold_e1(n, F1, gamma) - fold_e2(n, F2, GAMMA);
}

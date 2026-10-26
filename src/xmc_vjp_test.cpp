#include "PT_tensors.h"
#undef RES_FIT_H      // PT_tensors.h and res_fit.h share the include guard RES_FIT_H (a production quirk): include both
#include "res_fit.h"
#include "xmc_vjp_test.h"

#include <string>

#include "common_vars.h"
#include "matr.h"
#include "pt_vjp.h"

int xmc_vjp_selftest(PT_tensors & T, res_fit_data & RF, const char * blocks, int f){
    const int na = T.n_a, Nf = RF.N_fit;
    const size_t n2 = (size_t)na * na, n4 = n2 * n2, n6 = n4 * n2;
    if(f < 0 || f >= Nf) f = Nf / 2;
    fprintf(out_stream, "\nXMC VJP test: grid point %d of %d, E_fit = % .8f\n", f, Nf, RF.E_fit[f]);
    pt_vjp_shift = RF.E_fit[f];
    const double E_fit_f = RF.E_fit[f];
    pt_vjp_forward_override = [&](PT_tensors & TT, const std::string & blk){
        RF.E_fit[f] = pt_vjp_shift;       // the forward tables honour the (possibly perturbed) shift
        set_zero_matr(RF.RF_P3_JK, n6 * Nf); set_zero_matr(RF.RF_P3_AB, n6 * Nf);
        set_zero_matr(RF.RF_PV_JK, n4 * Nf); set_zero_matr(RF.RF_PV_AB, n4 * Nf);
        set_zero_matr(RF.RF_PH, n2 * Nf); set_zero_matr(RF.RF_PS, Nf);
        if(blk == "ALL"){ RF.calc_RF_2_CCVV(); RF.calc_RF_2_CAVV(); RF.calc_RF_2_AAVV(); RF.calc_RF_2_CCAV(); RF.calc_RF_2_CCAA(); RF.calc_RF_2_CV(); RF.calc_RF_2_AV(); RF.calc_RF_2_CA(); }
        else if(blk == "CCVV") RF.calc_RF_2_CCVV();
        else if(blk == "CAVV") RF.calc_RF_2_CAVV();
        else if(blk == "AAVV") RF.calc_RF_2_AAVV();
        else if(blk == "CCAV") RF.calc_RF_2_CCAV();
        else if(blk == "CCAA") RF.calc_RF_2_CCAA();
        else if(blk == "CV") RF.calc_RF_2_CV();
        else if(blk == "AV") RF.calc_RF_2_AV();
        else if(blk == "CA") RF.calc_RF_2_CA();
        else { fprintf(out_stream, "xmc_vjp_selftest: block %s is not available\n", blk.c_str()); return; }
        TT.RF_PS = RF.RF_PS[f];
        for(size_t i = 0; i < n2; i++) TT.RF_PH[i] = RF.RF_PH[i * Nf + f];
        for(size_t i = 0; i < n4; i++){ TT.RF_PV_JK[i] = RF.RF_PV_JK[i * Nf + f]; TT.RF_PV_AB[i] = RF.RF_PV_AB[i * Nf + f]; }
        for(size_t i = 0; i < n6; i++) TT.RF_P3_JK[i] = RF.RF_P3_JK[i * Nf + f];
        // RF_P3_AB: the EE kernels store [t,u,x,v,w,y] and permute at the end of E2_calc_EE ("order differs with the original XMCQDPT code"); res_fit_data keeps [t,u,v,w,x,y]
        for(int t = 0; t < na; t++) for(int u = 0; u < na; u++) for(int v = 0; v < na; v++) for(int w = 0; w < na; w++) for(int x = 0; x < na; x++) for(int y = 0; y < na; y++)
            TT.RF_P3_AB[((((t * na + u) * na + x) * na + v) * na + w) * na + y] = RF.RF_P3_AB[(((((t * na + u) * na + v) * na + w) * na + x) * na + y) * Nf + f];
    };
    const int fails = pt_vjp_selftest(T, blocks);
    RF.E_fit[f] = E_fit_f;
    pt_vjp_forward_override = nullptr;
    pt_vjp_shift = 0.0;
    return fails;
}

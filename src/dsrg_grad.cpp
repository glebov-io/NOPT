#include "dsrg_grad.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "aldet.h"
#include "aldet_rdm_adj.h"
#include "common_vars.h"
#include "integral_derivatives.h"
#include "inp_par_read.h"
#include "pt_canon.h"
#include "pt_chain.h"
#include "tensor_rotate.h"
#include "tensor_tape.h"
#include <omp.h>
#include "gradient_driver.h"

using tt::P;
using tt::Reg;

int dsrg_tape_energy(const DsrgTapeIn & in, DsrgTapeLedger * out, DsrgTapeAdj * adj, const DsrgRelaxSeeds * rs, DsrgRelaxOut * ro){
    const int nc = in.nc, na = in.na, nv = in.nv, nx = (int)in.naux;
    const double s = in.s;
    const bool grad = adj != nullptr;
    tt::Tape tp;
    // ---- leaves ----
    P VC = tp.leaf({nv, nc, nx}, in.VC, grad), VA = tp.leaf({nv, na, nx}, in.VA, grad), CA = tp.leaf({nc, na, nx}, in.CA, grad), AA = tp.leaf({na, na, nx}, in.AA, grad);
    P ec = tp.leaf({nc}, in.ec, grad), ea = tp.leaf({na}, in.ea, grad), ev = tp.leaf({nv}, in.ev, grad);
    P fca = tp.leaf({nc, na}, in.fca, grad), fcv = tp.leaf({nc, nv}, in.fcv, grad), fav = tp.leaf({na, nv}, in.fav, grad);
    P L1 = tp.leaf({na, na}, in.L1, grad), GAM = tp.leaf({na, na, na, na}, in.GAMMA, grad), G3 = tp.leaf({na, na, na, na, na, na}, in.G3, grad);
    P hcd = tp.leaf({nc}, in.hcd, grad), hact = tp.leaf({na, na}, in.hact, grad);
    std::vector<double> two_delta((size_t)na * na, 0.0); for(int t = 0; t < na; t++) two_delta[(size_t)t * na + t] = 2.0;
    P I2 = tp.leaf({na, na}, two_delta.data(), false);
    const double p = 1.0, m = -1.0;   // eps signs
    auto R4 = [&](const P & x, Reg k, P e0, double s0, P e1, double s1, P e2, double s2, P e3, double s3){ return tp.reg(x, k, s, {e0, e1, e2, e3}, {s0, s1, s2, s3}); };

    // ---- raw four-index integrals (chemist, from the RI B-tensors) ----
    P rCACA = tp.auxgemm(CA, CA);   // [i,u,j,v] = (iu|jv)
    P rCAVA = tp.auxgemm(CA, VA);   // [m,v,e,x] = (mv|ex)
    P rCAAA = tp.auxgemm(CA, AA);   // [m,u,v,y] = (mu|vy)
    P rVAAA = tp.auxgemm(VA, AA);   // [e,w,v,x] = (ew|vx)
    P rAAVC = tp.auxgemm(AA, VC);   // [u,v,a,i] = (uv|ai)
    P rAAAA = tp.auxgemm(AA, AA);   // [u,x,v,y] = (ux|vy)

    // ---- renormalized integrals and amplitudes (build_vt, build_amplitudes, build_aavv_tile) ----
    P Vt_aacc = R4(tp.einsum("iwjx->wxij", {rCACA}), Reg::ONE_PLUS_Q, ea, p, ea, p, ec, m, ec, m);
    P Vt_avca = R4(tp.einsum("mvex->vemx", {rCAVA}), Reg::ONE_PLUS_Q, ea, p, ev, p, ec, m, ea, m);
    P Vt_aaca = R4(tp.einsum("muvy->uvmy", {rCAAA}), Reg::ONE_PLUS_Q, ea, p, ea, p, ec, m, ea, m);
    P Vt_vaaa = R4(tp.einsum("ewvx->evwx", {rVAAA}), Reg::ONE_PLUS_Q, ev, p, ea, p, ea, m, ea, m);
    P Vt_avac = R4(tp.einsum("uxem->uexm", {rAAVC}), Reg::ONE_PLUS_Q, ea, p, ev, p, ea, m, ec, m);
    P T2_ccaa = R4(tp.einsum("iujv->ijuv", {rCACA}), Reg::R, ec, p, ec, p, ea, m, ea, m);
    P T2_caav = R4(tp.einsum("iuav->ivua", {rCAVA}), Reg::R, ec, p, ea, p, ea, m, ev, m);
    P T2_acav = R4(tp.einsum("uvai->viua", {rAAVC}), Reg::R, ea, p, ec, p, ea, m, ev, m);
    P T2_aava = R4(tp.einsum("euyv->uvey", {rVAAA}), Reg::R, ea, p, ea, p, ev, m, ea, m);
    P T2_caaa = R4(tp.einsum("iuvw->iwuv", {rCAAA}), Reg::R, ec, p, ea, p, ea, m, ea, m);

    // ---- singles: source s = f + sum_ux 1/2 (e_x - e_u) L1[x,u] (2 t - t_swap); T1 = s R(D), Ft = f + s q(D) ----
    P W = tp.lin({{1.0, tp.einsum("x,xu->xu", {ea, L1}, 0.5)}, {1.0, tp.einsum("u,xu->xu", {ea, L1}, -0.5)}});
    P src_cv = tp.lin({{1.0, fcv}, {2.0, tp.einsum("xu,uixa->ia", {W, T2_acav})}, {-1.0, tp.einsum("xu,iuxa->ia", {W, T2_caav})}});
    P src_ca = tp.lin({{1.0, fca}, {2.0, tp.einsum("xu,iuwx->iw", {W, T2_caaa})}, {-1.0, tp.einsum("xu,iuxw->iw", {W, T2_caaa})}});
    P src_av = tp.lin({{1.0, fav}, {2.0, tp.einsum("xu,vuax->va", {W, T2_aava})}, {-1.0, tp.einsum("xu,uvax->va", {W, T2_aava})}});
    P T1_cv = tp.reg(src_cv, in.ccvv_zero ? Reg::INV : Reg::R, s, {ec, ev}, {p, m});
    P T1_ca = tp.reg(src_ca, Reg::R, s, {ec, ea}, {p, m});
    P T1_av = tp.reg(src_av, Reg::R, s, {ea, ev}, {p, m});
    P Ft_cv = tp.lin({{1.0, fcv}, {1.0, tp.reg(src_cv, Reg::Q, s, {ec, ev}, {p, m})}});
    P Ft_ca = tp.lin({{1.0, fca}, {1.0, tp.reg(src_ca, Reg::Q, s, {ec, ea}, {p, m})}});
    P Ft_av = tp.lin({{1.0, fav}, {1.0, tp.reg(src_av, Reg::Q, s, {ea, ev}, {p, m})}});

    // ---- densities (set_densities): Eta1 = 2 - L1, physicist G2[p,q,r,s] = GAMMA[p,r,q,s], cumulant L2 ----
    P Eta = tp.lin({{1.0, I2}, {-1.0, L1}});
    P G2p = tp.einsum("prqs->pqrs", {GAM});
    P L2 = tp.lin({{1.0, G2p}, {-1.0, tp.einsum("pr,qs->pqrs", {L1, L1})}, {0.5, tp.einsum("ps,qr->pqrs", {L1, L1})}});

    // ---- [F,T1], [F,T2], [V,T1] ----
    P e_FT1 = tp.lin({{2.0, tp.einsum("ia,ia->", {Ft_cv, T1_cv})}, {2.0, tp.einsum("mu,mu->", {Ft_ca, T1_ca})},
                      {1.0, tp.einsum("vu,va,ua->", {L1, Ft_av, T1_av})}, {-1.0, tp.einsum("vu,mu,mv->", {L1, Ft_ca, T1_ca})}});
    P e_FT2 = tp.lin({{1.0, tp.einsum("xyuv,xe,uvey->", {L2, Ft_av, T2_aava})}, {-1.0, tp.einsum("xyuv,mv,muyx->", {L2, Ft_ca, T2_caaa})}});
    P e_VT1 = tp.lin({{1.0, tp.einsum("xyuv,evxy,ue->", {L2, Vt_vaaa, T1_av})}, {-1.0, tp.einsum("xyuv,uvmy,mx->", {L2, Vt_aaca, T1_ca})}});

    // ---- AAVV: 0.25 (S2.Vt) L1 L1 + 0.5 (T2.Vt) L2 ----
    // the virtual pair (e,f) is summed batch by batch outside the tape (dsrg_batch_AAVV): MT = sum_ef t2 vt, MS = sum_ef s2 vt are leaves
    std::vector<double> MT_v, MS_v;
    dsrg_batch_AAVV(in, &MT_v, &MS_v, nullptr, nullptr, nullptr, nullptr, nullptr);
    P MT = tp.leaf({na, na, na, na}, MT_v.data(), grad), MS = tp.leaf({na, na, na, na}, MS_v.data(), grad);
    P e_aavv1 = tp.einsum("uvwx,xv,wu->", {MS, L1, L1}, 0.25);
    P e_aavv2 = tp.einsum("uvwx,uvwx->", {MT, L2}, 0.5);
    // ---- CCAA: 0.25 (Vt.S2) Eta Eta + 0.5 (Vt.T2) L2 ----
    P S2_ccaa = tp.lin({{2.0, T2_ccaa}, {-1.0, tp.einsum("ijvu->ijuv", {T2_ccaa})}});
    P NT = tp.einsum("wxij,ijuv->wxuv", {Vt_aacc, T2_ccaa});
    P NS = tp.einsum("wxij,ijuv->wxuv", {Vt_aacc, S2_ccaa});
    P e_ccaa1 = tp.einsum("wxuv,vx,uw->", {NS, Eta, Eta}, 0.25);
    P e_ccaa2 = tp.einsum("wxuv,uvwx->", {NT, L2}, 0.5);
    // ---- CAAV ----
    P QB = tp.einsum("pmqe->empq", {T2_acav});                 // T2_acav[p,m,q,e]
    P QC = tp.einsum("mpqe->empq", {T2_caav});                 // T2_caav[m,p,q,e]
    P R1 = tp.lin({{2.0, QC}, {-1.0, QB}});                     // S2_caav[m,y,u,e]
    P QA = tp.lin({{2.0, QB}, {-1.0, QC}});                     // S2_acav[y,m,u,e] (= R2)
    P C12 = tp.einsum("vemx,emyu->vxyu", {Vt_avca, R1});
    tp.einsum_into(C12, "vexm,emyu->vxyu", {Vt_avac, QA});
    P e_caav1 = tp.einsum("vxyu,uv,xy->", {C12, Eta, L1}, 0.5);
    P RA = tp.einsum("uexm,emvy->uxvy", {Vt_avac, QA});
    P RB = tp.einsum("uemx,emvy->uxvy", {Vt_avca, QB});
    P RC = tp.einsum("vemx,emuy->vxuy", {Vt_avca, QC});
    P e_caav2 = tp.lin({{1.0, tp.einsum("uxvy,uvxy->", {RA, L2})}, {-1.0, tp.einsum("uxvy,uvxy->", {RB, L2})}, {-1.0, tp.einsum("vxuy,uvxy->", {RC, L2})}});
    // ---- CAAA ----
    P e_caaa1, e_caaa2;
    {
        P Lm = tp.einsum("vzmx,wz->vxmw", {Vt_aaca, Eta});
        P Rr = tp.lin({{2.0, tp.einsum("myuw->mwyu", {T2_caaa})}, {-1.0, tp.einsum("mywu->mwyu", {T2_caaa})}});
        P G = tp.einsum("vxmw,mwyu->vxyu", {Lm, Rr});
        e_caaa1 = tp.einsum("vxyu,uv,xy->", {G, Eta, L1}, 0.25);
        P A1 = tp.einsum("uvmw,wz->uvmz", {Vt_aaca, L1});
        P t1 = tp.einsum("uvmz,mzxy->uvxy", {A1, T2_caaa});
        P L2f = tp.einsum("wumx,wz->uxmz", {Vt_aaca, Eta});
        P L3f = tp.einsum("uwmx,wz->uxmz", {Vt_aaca, Eta});
        P Rs2 = tp.lin({{2.0, tp.einsum("mvzy->mzvy", {T2_caaa})}, {-1.0, tp.einsum("mvyz->mzvy", {T2_caaa})}});
        P G2t = tp.einsum("uxmz,mzvy->uxvy", {L2f, Rs2});
        P G3t = tp.einsum("uxmz,mvzy->uxvy", {L3f, T2_caaa});
        P G4t = tp.einsum("vxmz,muyz->vxuy", {L3f, T2_caaa});
        e_caaa2 = tp.lin({{0.5, tp.einsum("uvxy,uvxy->", {t1, L2})}, {0.5, tp.einsum("uxvy,uvxy->", {G2t, L2})},
                          {-0.5, tp.einsum("uxvy,uvxy->", {G3t, L2})}, {-0.5, tp.einsum("vxuy,uvxy->", {G4t, L2})}});
    }
    // ---- AAAV ----
    P e_aaav1, e_aaav2;
    {
        P Lm = tp.einsum("evwx,wz->vxez", {Vt_vaaa, L1});
        P Rr = tp.lin({{2.0, tp.einsum("zyeu->ezyu", {T2_aava})}, {-1.0, tp.einsum("yzeu->ezyu", {T2_aava})}});
        P G = tp.einsum("vxez,ezyu->vxyu", {Lm, Rr});
        e_aaav1 = tp.einsum("vxyu,uv,xy->", {G, Eta, L1}, 0.25);
        P Rr1 = tp.einsum("ezxy,wz->ewxy", {Vt_vaaa, Eta});
        P t1 = tp.einsum("uvew,ewxy->uvxy", {T2_aava, Rr1});
        P L2f = tp.einsum("euwx,wz->uxez", {Vt_vaaa, L1});
        P L3f = tp.einsum("euxw,wz->uxez", {Vt_vaaa, L1});
        P Rs2 = tp.lin({{2.0, tp.einsum("zvey->ezvy", {T2_aava})}, {-1.0, tp.einsum("vzey->ezvy", {T2_aava})}});
        P G2t = tp.einsum("uxez,ezvy->uxvy", {L2f, Rs2});
        P G3t = tp.einsum("uxez,zvey->uxvy", {L3f, T2_aava});
        P G4t = tp.einsum("vxez,uzey->vxuy", {L3f, T2_aava});
        e_aaav2 = tp.lin({{0.5, tp.einsum("uvxy,uvxy->", {t1, L2})}, {0.5, tp.einsum("uxvy,uvxy->", {G2t, L2})},
                          {-0.5, tp.einsum("uxvy,uvxy->", {G3t, L2})}, {-0.5, tp.einsum("vxuy,uvxy->", {G4t, L2})}});
    }
    // ---- lambda3: the cumulant of the 3-body moment (compute_e3), closed against Vt T2 ----
    P L3 = tp.lin({{1.0, G3}});
    {
        const char * one_two[9][2] = {{"ps,qrtu->pqrstu", "-1"}, {"qt,prsu->pqrstu", "-1"}, {"ru,pqst->pqrstu", "-1"},
                                      {"pt,qrsu->pqrstu", "0.5"}, {"pu,qrts->pqrstu", "0.5"}, {"qs,prtu->pqrstu", "0.5"},
                                      {"qu,prst->pqrstu", "0.5"}, {"rs,pqut->pqrstu", "0.5"}, {"rt,pqsu->pqrstu", "0.5"}};
        for(auto & t : one_two) tp.einsum_into(L3, t[0], {L1, G2p}, atof(t[1]));
        const char * three[6][2] = {{"ps,qt,ru->pqrstu", "2"}, {"ps,qu,rt->pqrstu", "-1"}, {"pu,qt,rs->pqrstu", "-1"},
                                    {"pt,qs,ru->pqrstu", "-1"}, {"pt,qu,rs->pqrstu", "0.5"}, {"pu,qs,rt->pqrstu", "0.5"}};
        for(auto & t : three) tp.einsum_into(L3, t[0], {L1, L1, L1}, atof(t[1]));
    }
    P M3v = tp.einsum("ewxy,uvez->wxyuvz", {Vt_vaaa, T2_aava});
    P e_3v = tp.einsum("wxyuvz,xyzuwv->", {M3v, L3});
    P M3c = tp.einsum("mwxy,uvmz->wxyuvz", {T2_caaa, Vt_aaca});
    P e_3c = tp.einsum("wxyuvz,xyzuwv->", {M3c, L3}, -1.0);
    // ---- reference energy (compute_eref) ----
    P e_ref = tp.lin({{1.0, tp.einsum("m->", {hcd})}, {1.0, tp.einsum("m->", {ec})}, {0.5, tp.einsum("uv,vu->", {hact, L1})},
                      {0.5, tp.einsum("u,uu->", {ea, L1})}, {0.5, tp.einsum("uxvy,xyuv->", {rAAAA, L2})}});
    // ---- DF CCVV, CAVV, CCAV (dsrg_sf_batch): computed batch by batch outside the tape (dsrg_batch_vjp.cpp, never materialized); on the tape they
    //      are leaves -- the CCVV energy and the two n_a x n_a intermediates C, closed here with L1 and Eta and folded into the dressed operator ----
    std::vector<double> Ccavv_v, Cccav_v;
    dsrg_batch_CAVV(in, &Ccavv_v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    dsrg_batch_CCAV(in, &Cccav_v, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    const double eccvv_v = dsrg_batch_CCVV(in, 0.0, nullptr, nullptr, nullptr);
    P e_ccvv = tp.leaf({}, &eccvv_v, grad);
    P C_cavv = tp.leaf({na, na}, Ccavv_v.data(), grad), C_ccav = tp.leaf({na, na}, Cccav_v.data(), grad);
    P e_cavv = tp.einsum("vu,uv->", {C_cavv, L1});
    P e_ccav = tp.einsum("vu,uv->", {C_ccav, Eta});
    P total = tp.lin({{1.0, e_ref}, {1.0, e_FT1}, {1.0, e_FT2}, {1.0, e_VT1}, {1.0, e_aavv1}, {1.0, e_aavv2}, {1.0, e_ccaa1}, {1.0, e_ccaa2},
                      {1.0, e_caav1}, {1.0, e_caav2}, {1.0, e_caaa1}, {1.0, e_caaa2}, {1.0, e_aaav1}, {1.0, e_aaav2}, {1.0, e_3v}, {1.0, e_3c},
                      {1.0, e_ccvv}, {1.0, e_cavv}, {1.0, e_ccav}});
    // ---- relax=once: the dressed active operator Hbar = seed + 1/2 [H~1, A] (build_hbar) with the batched CAVV/CCAV Hbar1 folds of the driver,
    //      de-normal-ordered (degno) to e0_d + h1_d + h2_d (chemist). Every term below carries the production comment it transcribes. ----
    P e0d, h1d, h2d, Hb1, Hb2;
    if(rs || ro){
        P Vt_avaa = R4(tp.einsum("zxay->zaxy", {tp.auxgemm(AA, VA)}), Reg::ONE_PLUS_Q, ea, p, ev, p, ea, m, ea, m);   // <za|xy> = (zx|ay)
        P S2_caaa = tp.lin({{2.0, T2_caaa}, {-1.0, tp.einsum("iwvu->iwuv", {T2_caaa})}});
        P S2_caav = tp.lin({{2.0, T2_caav}, {-1.0, tp.einsum("viua->ivua", {T2_acav})}});
        P S2_acav = tp.lin({{2.0, T2_acav}, {-1.0, tp.einsum("ivua->viua", {T2_caav})}});
        P S2_aava = tp.lin({{2.0, T2_aava}, {-1.0, tp.einsum("vuey->uvey", {T2_aava})}});
        P G2_avac = tp.lin({{2.0, Vt_avac}, {-1.0, tp.einsum("uemv->uevm", {Vt_avca})}});     // 2<ue|vm> - <ue|mv>
        P G2_avaa = tp.lin({{2.0, Vt_avaa}, {-1.0, tp.einsum("ueyx->uexy", {Vt_avaa})}});
        P G2_aaac = tp.lin({{2.0, tp.einsum("vumw->uvwm", {Vt_aaca})}, {-1.0, tp.einsum("uvmw->uvwm", {Vt_aaca})}});
        P Fav = Ft_av, Fca = Ft_ca, Fcv = Ft_cv;
        // ---- C1 (temp1[u,v]) ----
        P t1 = tp.zeros({na, na});
        // H1_T_C1a_smallS
        tp.einsum_into(t1, "ve,ue->uv", {Fav, T1_av});
        tp.einsum_into(t1, "mu,mv->uv", {Fca, T1_ca}, -1.0);
        tp.einsum_into(t1, "me,umve->uv", {Fcv, S2_acav});
        tp.einsum_into(t1, "mx,muxv->uv", {Fca, S2_caaa});
        tp.einsum_into(t1, "xe,yuev,xy->uv", {Fav, S2_aava, L1}, 0.5);
        tp.einsum_into(t1, "my,muxv,xy->uv", {Fca, S2_caaa, L1}, -0.5);
        // H2_T_C1a_smallG
        tp.einsum_into(t1, "ma,uavm->uv", {T1_cv, G2_avac});
        tp.einsum_into(t1, "ma,uavm->uv", {T1_ca, G2_aaac});
        tp.einsum_into(t1, "xe,yx,uevy->uv", {T1_av, L1, G2_avaa}, 0.5);
        tp.einsum_into(t1, "mx,xy,uyvm->uv", {T1_ca, L1, G2_aaac}, -0.5);
        tp.einsum_into(t1, "wezx,uvey,xyuv->wz", {G2_avaa, T2_aava, L2}, 0.5);
        tp.einsum_into(t1, "wuzm,mvxy,xyuv->wz", {G2_aaac, T2_caaa, L2}, -0.5);
        // H2_T_C1a_smallS direct terms
        tp.einsum_into(t1, "uemz,mwue->wz", {Vt_avca, S2_caav});
        tp.einsum_into(t1, "uezm,wmue->wz", {Vt_avac, S2_acav});
        tp.einsum_into(t1, "wemu,muze->wz", {Vt_avca, S2_caav}, -1.0);
        tp.einsum_into(t1, "weum,umze->wz", {Vt_avac, S2_acav}, -1.0);
        tp.einsum_into(t1, "vumz,mwvu->wz", {Vt_aaca, S2_caaa});
        tp.einsum_into(t1, "ewvu,vuez->wz", {Vt_vaaa, S2_aava}, -1.0);
        // temp[wzuv] closed with L1[uv]
        {
            P tw = tp.zeros({na, na, na, na});
            tp.einsum_into(tw, "wvzu->wzuv", {MS}, 0.5);                // 0.5 S2[wvef] H2[efzu]
            tp.einsum_into(tw, "wvex,exzu->wzuv", {S2_aava, Vt_vaaa}, 0.5);
            tp.einsum_into(tw, "vwex,exuz->wzuv", {S2_aava, Vt_vaaa}, 0.5);
            tp.einsum_into(tw, "wmue,vezm->wzuv", {S2_acav, Vt_avac}, -0.5);
            tp.einsum_into(tw, "mwue,vemz->wzuv", {S2_caav, Vt_avca}, -0.5);
            tp.einsum_into(tw, "mwxu,xvmz->wzuv", {S2_caaa, Vt_aaca}, -0.5);
            tp.einsum_into(tw, "mwux,vxmz->wzuv", {S2_caaa, Vt_aaca}, -0.5);
            tp.einsum_into(tw, "jwxu,xy,yvjz->wzuv", {S2_caaa, L1, Vt_aaca}, 0.25);
            tp.einsum_into(tw, "ywbu,xy,bvxz->wzuv", {S2_aava, L1, Vt_vaaa}, -0.25);
            tp.einsum_into(tw, "wybu,xy,bvzx->wzuv", {S2_aava, L1, Vt_vaaa}, -0.25);
            tp.einsum_into(t1, "wzuv,uv->wz", {tw, L1});
        }
        // temp[wzuv] closed with Eta1[uv]
        {
            P S2_ccaa = tp.lin({{2.0, T2_ccaa}, {-1.0, tp.einsum("ijvu->ijuv", {T2_ccaa})}});
            P tw = tp.zeros({na, na, na, na});
            tp.einsum_into(tw, "mnzu,wvmn->wzuv", {S2_ccaa, Vt_aacc}, -0.5);
            tp.einsum_into(tw, "mxzu,wvmx->wzuv", {S2_caaa, Vt_aaca}, -0.5);
            tp.einsum_into(tw, "mxuz,vwmx->wzuv", {S2_caaa, Vt_aaca}, -0.5);
            tp.einsum_into(tw, "vmze,weum->wzuv", {S2_acav, Vt_avac}, 0.5);
            tp.einsum_into(tw, "mvze,wemu->wzuv", {S2_caav, Vt_avca}, 0.5);
            tp.einsum_into(tw, "xvez,ewxu->wzuv", {S2_aava, Vt_vaaa}, 0.5);
            tp.einsum_into(tw, "vxez,ewux->wzuv", {S2_aava, Vt_vaaa}, 0.5);
            tp.einsum_into(tw, "yvbz,xy,bwxu->wzuv", {S2_aava, Eta, Vt_vaaa}, -0.25);
            tp.einsum_into(tw, "jvxz,ywju,xy->wzuv", {S2_caaa, Vt_aaca, Eta}, 0.25);
            tp.einsum_into(tw, "jvzx,wyju,xy->wzuv", {S2_caaa, Vt_aaca, Eta}, 0.25);
            tp.einsum_into(t1, "wzuv,uv->wz", {tw, Eta});
        }
        // L2 ring terms
        tp.einsum_into(t1, "vujz,jwyx,xyuv->wz", {Vt_aaca, T2_caaa, L2}, 0.5);
        tp.einsum_into(t1, "auzx,wvay,xyuv->wz", {Vt_vaaa, S2_aava, L2}, 0.5);
        tp.einsum_into(t1, "auxz,wvay,xyuv->wz", {Vt_vaaa, T2_aava, L2}, -0.5);
        tp.einsum_into(t1, "auxz,vway,xyvu->wz", {Vt_vaaa, T2_aava, L2}, -0.5);
        tp.einsum_into(t1, "bwyx,vubz,xyuv->wz", {Vt_vaaa, T2_aava, L2}, -0.5);
        tp.einsum_into(t1, "wuix,ivzy,xyuv->wz", {Vt_aaca, S2_caaa, L2}, -0.5);
        tp.einsum_into(t1, "uwix,ivzy,xyuv->wz", {Vt_aaca, T2_caaa, L2}, 0.5);
        tp.einsum_into(t1, "uwix,ivyz,xyvu->wz", {Vt_aaca, T2_caaa, L2}, 0.5);
        tp.einsum_into(t1, "avxy,uwaz,xyuv->wz", {Vt_vaaa, S2_aava, L2}, 0.5);
        tp.einsum_into(t1, "uviy,iwxz,xyuv->wz", {Vt_aaca, S2_caaa, L2}, -0.5);
        // ---- C2 (C2t[u,v,x,y]) ladder / ring ----
        P c2 = tp.zeros({na, na, na, na});
        tp.einsum_into(c2, "uvxy->uvxy", {MT});                         // H2[efxy] T2[uvef]
        tp.einsum_into(c2, "uvmn,mnxy->uvxy", {Vt_aacc, T2_ccaa});
        tp.einsum_into(c2, "uvew,ewxy->uvxy", {T2_aava, Vt_vaaa});
        tp.einsum_into(c2, "vuew,ewyx->uvxy", {T2_aava, Vt_vaaa});
        tp.einsum_into(c2, "vumw,mwyx->uvxy", {Vt_aaca, T2_caaa});
        tp.einsum_into(c2, "uvmw,mwxy->uvxy", {Vt_aaca, T2_caaa});
        // particle/hole temp2, then C2t[uvxy] += t, C2t[vuyx] += t
        {
            P t2 = tp.zeros({na, na, na, na});
            tp.einsum_into(t2, "uvay,xa->uvxy", {T2_aava, Fav});
            tp.einsum_into(t2, "ivxy,iu->uvxy", {T2_caaa, Fca}, -1.0);
            tp.einsum_into(t2, "ua,avxy->uvxy", {T1_av, Vt_vaaa});
            tp.einsum_into(t2, "ix,uviy->uvxy", {T1_ca, Vt_aaca}, -1.0);
            tp.einsum_into(t2, "vuaw,wz,azyx->uvxy", {T2_aava, L1, Vt_vaaa}, -0.5);
            tp.einsum_into(t2, "izyx,vuiw,wz->uvxy", {T2_caaa, Vt_aaca, Eta}, -0.5);
            tp.einsum_into(t2, "uexm,vmye->uvxy", {Vt_avac, S2_acav});
            tp.einsum_into(t2, "wumx,mvwy->uvxy", {Vt_aaca, S2_caaa});
            tp.einsum_into(t2, "auwx,zvay,wz->uvxy", {Vt_vaaa, S2_aava, L1}, 0.5);
            tp.einsum_into(t2, "ivwy,zuix,wz->uvxy", {S2_caaa, Vt_aaca, L1}, -0.5);
            tp.einsum_into(t2, "uemx,vmye->uvxy", {Vt_avca, T2_acav}, -1.0);
            tp.einsum_into(t2, "uwmx,mvwy->uvxy", {Vt_aaca, T2_caaa}, -1.0);
            tp.einsum_into(t2, "auxw,zvay,wz->uvxy", {Vt_vaaa, T2_aava, L1}, -0.5);
            tp.einsum_into(t2, "ivwy,uzix,wz->uvxy", {T2_caaa, Vt_aaca, L1}, 0.5);
            tp.einsum_into(t2, "vemx,muye->uvxy", {Vt_avca, T2_caav}, -1.0);
            tp.einsum_into(t2, "vwmx,muyw->uvxy", {Vt_aaca, T2_caaa}, -1.0);
            tp.einsum_into(t2, "avxw,uzay,wz->uvxy", {Vt_vaaa, T2_aava, L1}, -0.5);
            tp.einsum_into(t2, "iuyw,vzix,wz->uvxy", {T2_caaa, Vt_aaca, L1}, 0.5);
            tp.einsum_into(c2, "uvxy->uvxy", {t2});
            tp.einsum_into(c2, "vuyx->uvxy", {t2});
        }
        // seed (bare F_act diagonal = eps_a, bare <uv|xy>) + symmetric-accumulation Hermitization; the driver's batched folds
        Hb1 = tp.lin({{1.0, tp.einsum("u,uv->uv", {ea, I2}, 0.5)}, {0.5, t1}, {0.5, tp.einsum("vu->uv", {t1})},
                      {0.5, C_cavv}, {0.5, tp.einsum("vu->uv", {C_cavv})}, {-0.5, C_ccav}, {-0.5, tp.einsum("vu->uv", {C_ccav})}});
        Hb2 = tp.lin({{1.0, tp.einsum("uxvy->uvxy", {rAAAA})}, {0.5, c2}, {0.5, tp.einsum("xyuv->uvxy", {c2})}});
        // degno: pair symmetrization, hbar = 2 H2 - H2^swap, scalar / 1-body folds, chemist re-pack
        P H2s = tp.lin({{0.5, Hb2}, {0.5, tp.einsum("vuyx->uvxy", {Hb2})}});
        P hb = tp.lin({{2.0, H2s}, {-1.0, tp.einsum("pqsr->pqrs", {H2s})}});
        P s1 = tp.einsum("vu,uv->", {Hb1, L1}, -1.0);
        P s2 = tp.lin({{0.25, tp.einsum("uv,vyux,xy->", {L1, hb, L1})}, {-0.5, tp.einsum("xyuv,uvxy->", {H2s, L2})}});
        e0d = tp.lin({{1.0, total}, {1.0, s1}, {1.0, s2}});
        h1d = tp.lin({{1.0, Hb1}, {-0.5, tp.einsum("uxvy,yx->uv", {hb, L1})}});
        h2d = tp.einsum("acbd->abcd", {H2s});
    }
    if(ro){
        ro->e0 = e0d->v[0] + in.escalar; ro->h1 = h1d->v; ro->h2_chem = h2d->v; ro->Hbar1 = Hb1->v; ro->Hbar2 = Hb2->v;
    }
    if(out){
        DsrgTapeLedger & L = *out;
        L.E_FT1 = e_FT1->v[0]; L.E_FT2 = e_FT2->v[0]; L.E_VT1 = e_VT1->v[0];
        L.aavv_L1 = e_aavv1->v[0]; L.aavv_L2 = e_aavv2->v[0]; L.ccaa_L1 = e_ccaa1->v[0]; L.ccaa_L2 = e_ccaa2->v[0];
        L.caav_L1 = e_caav1->v[0]; L.caav_L2 = e_caav2->v[0]; L.caaa_L1 = e_caaa1->v[0]; L.caaa_L2 = e_caaa2->v[0];
        L.aaav_L1 = e_aaav1->v[0]; L.aaav_L2 = e_aaav2->v[0]; L.E3v = e_3v->v[0]; L.E3c = e_3c->v[0]; L.Eref = e_ref->v[0] + in.escalar;
        L.ccvv = e_ccvv->v[0]; L.cavv = e_cavv->v[0]; L.ccav = e_ccav->v[0]; L.total = total->v[0] + in.escalar;
    }
    if(adj){
        if(rs){
            // relaxed energy: Hellmann-Feynman on the dressed root(s), phi = e0b e0_d + <h1b, h1_d> + <h2b, h2_d>
            P b1 = tp.leaf({na, na}, rs->h1b, false), b2 = tp.leaf({na, na, na, na}, rs->h2b_chem, false);
            P phi = tp.lin({{rs->e0b, e0d}, {1.0, tp.einsum("uv,uv->", {b1, h1d})}, {1.0, tp.einsum("abcd,abcd->", {b2, h2d})}});
            tp.backward(phi);
        }
        else tp.backward(total);
        adj->VC = VC->g; adj->VA = VA->g; adj->CA = CA->g; adj->AA = AA->g; adj->ec = ec->g; adj->ea = ea->g; adj->ev = ev->g;
        // the batched DF terms: their seeds from the tape, the adjoints by a second batched sweep
        dsrg_batch_CCVV(in, e_ccvv->g[0], &adj->VC, &adj->ec, &adj->ev);
        dsrg_batch_AAVV(in, nullptr, nullptr, MT->g.data(), MS->g.data(), &adj->VA, &adj->ea, &adj->ev);
        dsrg_batch_CAVV(in, nullptr, C_cavv->g.data(), &adj->VC, &adj->VA, &adj->ec, &adj->ea, &adj->ev);
        dsrg_batch_CCAV(in, nullptr, C_ccav->g.data(), &adj->VC, &adj->CA, &adj->ec, &adj->ea, &adj->ev);
        adj->fca = fca->g; adj->fcv = fcv->g; adj->fav = fav->g; adj->L1 = L1->g; adj->GAMMA = GAM->g; adj->G3 = G3->g; adj->hcd = hcd->g; adj->hact = hact->g;
    }
    return 0;
}

int dsrg_tape_selftest(const DsrgTapeIn & in){
    DsrgTapeLedger L0; DsrgTapeAdj A;
    dsrg_tape_energy(in, &L0, &A);
    std::mt19937_64 gen(31337);
    auto rnd = [&](){ return std::uniform_real_distribution<double>(-1.0, 1.0)(gen); };
    struct G{ const char * name; const double * x; const std::vector<double> * g; };
    std::vector<G> groups = {{"B(VC)", in.VC, &A.VC}, {"B(VA)", in.VA, &A.VA}, {"B(CA)", in.CA, &A.CA}, {"B(AA)", in.AA, &A.AA},
                             {"eps_c", in.ec, &A.ec}, {"eps_a", in.ea, &A.ea}, {"eps_v", in.ev, &A.ev}, {"f_ca", in.fca, &A.fca}, {"f_cv", in.fcv, &A.fcv},
                             {"f_av", in.fav, &A.fav}, {"L1", in.L1, &A.L1}, {"GAMMA", in.GAMMA, &A.GAMMA}, {"G3", in.G3, &A.G3},
                             {"h_core", in.hcd, &A.hcd}, {"h_act", in.hact, &A.hact}};
    int fails = 0;
    fprintf(out_stream, "\nDSRG tape adjoint test (dot products against 4-point central differences of the tape energy):\n");
    for(auto & gr : groups){
        const size_t n = gr.g->size();
        if(n == 0) continue;
        std::vector<double> dx(n); for(double & v : dx) v = rnd();
        double ana = 0; for(size_t i = 0; i < n; i++) ana += (*gr.g)[i] * dx[i];
        double xs = 0; for(size_t i = 0; i < n; i++) xs = std::max(xs, std::fabs(gr.x[i]));
        const double h = 1e-4 * std::max(1.0, xs);
        auto at = [&](double k){
            std::vector<double> x(gr.x, gr.x + n);
            for(size_t i = 0; i < n; i++) x[i] += k * h * dx[i];
            DsrgTapeIn in2 = in;
            const double ** slot = nullptr;
            if(gr.x == in.VC) slot = &in2.VC; else if(gr.x == in.VA) slot = &in2.VA; else if(gr.x == in.CA) slot = &in2.CA; else if(gr.x == in.AA) slot = &in2.AA;
            else if(gr.x == in.ec) slot = &in2.ec; else if(gr.x == in.ea) slot = &in2.ea; else if(gr.x == in.ev) slot = &in2.ev;
            else if(gr.x == in.fca) slot = &in2.fca; else if(gr.x == in.fcv) slot = &in2.fcv; else if(gr.x == in.fav) slot = &in2.fav;
            else if(gr.x == in.L1) slot = &in2.L1; else if(gr.x == in.GAMMA) slot = &in2.GAMMA; else if(gr.x == in.G3) slot = &in2.G3;
            else if(gr.x == in.hcd) slot = &in2.hcd; else slot = &in2.hact;
            *slot = x.data();
            DsrgTapeLedger L; dsrg_tape_energy(in2, &L, nullptr);
            return L.total;
        };
        const double fd = (-at(2) + 8 * at(1) - 8 * at(-1) + at(-2)) / (12 * h);
        const double err = std::fabs(ana - fd) / std::max(1e-10, std::max(std::fabs(ana), std::fabs(fd)));
        const double noise = 1e-14 * std::fabs(L0.total) / h;     // rounding of the total energy in the difference quotient
        const bool ok = err < 1e-6 || std::fabs(ana - fd) < noise;
        if(!ok) fails++;
        fprintf(out_stream, "   %-7s analytic % .12e  FD % .12e  rel.err %.1e  %s\n", gr.name, ana, fd, err, ok ? "ok" : "MISMATCH");
    }
    return fails;
}

// ---- nuclear gradient ---------------------------------------------------------------------------------------------------------------------
DsrgGradRequest g_dsrg_grad;

int dsrg_grad_snapshot(molecule * M, dsrg_par * dsrg){
    if(!g_dsrg_grad.want) return 0;
    g_dsrg_grad.status = 0; g_dsrg_grad.done = false;
    if(dsrg->relax != DSRG_RELAX_NONE && dsrg->relax != DSRG_RELAX_ONCE){
        fprintf(out_stream, "ERROR: the analytic DSRG-PT2 gradient supports relax=none and relax=once\n");
        g_dsrg_grad.status = 1; return 1;
    }
    if(dsrg->cas->ci_solver != CISOLVER_ALDET){
        fprintf(out_stream, "ERROR: the analytic DSRG-PT2 gradient needs the determinant CI (cisolver=aldet)\n");
        g_dsrg_grad.status = 1; return 1;
    }
    delete g_dsrg_grad.ref;
    g_dsrg_grad.ref = new CasRef;
    if(g_dsrg_grad.ref->build(M)){ g_dsrg_grad.status = 1; return 1; }
    if(cas_reference_check(*g_dsrg_grad.ref, "DSRG-PT2 gradient")){ g_dsrg_grad.status = 1; return 1; }
    return 0;
}

namespace{
// M_ik += sum over every leg of X: Xbar[.., i, ..] X[.., k, ..]  (the derivative of <Xbar, X> under X -> rotated legs, X' = (1 + Omega) X per leg)
void leg_moment(int na, int rank, const double * Xb, const double * X, double * Mx){
    size_t tot = 1; for(int r = 0; r < rank; r++) tot *= (size_t)na;
    for(int leg = 0; leg < rank; leg++){
        size_t st = 1; for(int r = leg + 1; r < rank; r++) st *= (size_t)na;
        for(size_t idx = 0; idx < tot; idx++){
            const double xb = Xb[idx]; if(xb == 0.0) continue;
            const int i = (int)((idx / st) % na);
            const size_t base = idx - (size_t)i * st;
            for(int k = 0; k < na; k++) Mx[(size_t)i * na + k] += xb * X[base + (size_t)k * st];
        }
    }
}
}

int dsrg_gradient(molecule * M, const DsrgTapeIn & in, const double * Csc, const double * eps, const double * Uc, const double * Ua, const double * Uv,
                  const std::vector<int> & roots, const std::vector<double> & wts, const DsrgRelaxSeeds * rs, double E_relax){
    if(!g_dsrg_grad.want || g_dsrg_grad.status) return g_dsrg_grad.status;
    if(!g_dsrg_grad.ref){ fprintf(out_stream, "ERROR: no reference snapshot for the DSRG-PT2 gradient\n"); g_dsrg_grad.status = 1; return 1; }
    const CasRef & R = *g_dsrg_grad.ref;
    const int n = R.n, nc = R.nc, na = R.na, nv = R.nv, nmo = R.nmo, naux = R.naux;
    if(in.nc != nc || in.na != na || in.nv != nv || in.naux != naux){ fprintf(out_stream, "ERROR: DSRG dimensions do not match the CAS-SCF snapshot\n"); g_dsrg_grad.status = 1; return 1; }
    const size_t a2 = (size_t)na * na, a4 = a2 * a2, a6 = a4 * a2;

    // the semicanonical orbitals must be the block rotations of the snapshot orbitals
    {
        double dev = 0;
        const int first[3] = {0, nc, nc + na}, dim[3] = {nc, na, nv};
        const double * U[3] = {Uc, Ua, Uv};
        for(int b = 0; b < 3; b++) for(int i = 0; i < dim[b]; i++) for(int a = 0; a < n; a++){
            double v = 0; for(int k = 0; k < dim[b]; k++) v += U[b][(size_t)k * dim[b] + i] * R.C0[(size_t)(first[b] + k) * n + a];
            dev = std::max(dev, std::fabs(v - Csc[(size_t)(first[b] + i) * n + a]));
        }
        fprintf(out_stream, "DSRG-PT2 gradient: semicanonical orbitals vs block rotations of the CAS-SCF orbitals: %.2e\n", dev);
        if(dev > 1e-8){ fprintf(out_stream, "ERROR: the semicanonical orbitals are not the block rotations of the CAS-SCF orbitals\n"); g_dsrg_grad.status = 1; return 1; }
    }

    double tw = omp_get_wtime();
    auto lap = [&](const char * what){ const double t = omp_get_wtime(); fprintf(out_stream, "DSRG-PT2 gradient: %-40s %8.1f s   (memory %.0f MB, peak %.0f MB)\n", what, t - tw, grad_rss_mb(), grad_peak_rss_mb()); fflush(out_stream); tw = t; };
    // 1. the energy and the adjoints of its semicanonical inputs
    DsrgTapeLedger L; DsrgTapeAdj A;
    dsrg_tape_energy(in, &L, &A, rs);
    g_dsrg_grad.E = rs ? E_relax : L.total;

    lap(rs ? "tape (energy, dressed operator, adjoints)" : "tape (energy and adjoints)");
    // 2. RI B-tensors -> orbitals, (ab|P), (P|Q)
    PTChainOut out; out.resize(nmo, n, naux);
    PTInputAdj pa; pa.VC = A.VC.data(); pa.VA = A.VA.data(); pa.CA = A.CA.data(); pa.AA = A.AA.data();
    pt_chain_B(n, naux, nc, na, nv, Csc, R.L(), R.V.data(), pa, &out, &R.ri());
    // 3. bare h: active block and core diagonal in the semicanonical orbitals
    {
        std::vector<double> Hc((size_t)nmo * n, 0.0);     // rows: H c_p
        for(int p = 0; p < nc + na; p++) for(int a = 0; a < n; a++){ double v = 0; for(int b = 0; b < n; b++) v += R.h[(size_t)a * n + b] * Csc[(size_t)p * n + b]; Hc[(size_t)p * n + a] = v; }
        for(int u = 0; u < na; u++) for(int v = 0; v < na; v++){
            const double hb = A.hact[(size_t)u * na + v]; if(hb == 0.0) continue;
            const double * cu = Csc + (size_t)(nc + u) * n, * cv = Csc + (size_t)(nc + v) * n;
            for(int a = 0; a < n; a++){
                out.Cbar[(size_t)(nc + u) * n + a] += hb * Hc[(size_t)(nc + v) * n + a];
                out.Cbar[(size_t)(nc + v) * n + a] += hb * Hc[(size_t)(nc + u) * n + a];
                for(int b = 0; b < n; b++) out.hbar[(size_t)a * n + b] += 0.5 * hb * (cu[a] * cv[b] + cv[a] * cu[b]);
            }
        }
        for(int i = 0; i < nc; i++){
            const double hb = A.hcd[i]; if(hb == 0.0) continue;
            const double * ci = Csc + (size_t)i * n;
            for(int a = 0; a < n; a++){
                out.Cbar[(size_t)i * n + a] += 2.0 * hb * Hc[(size_t)i * n + a];
                for(int b = 0; b < n; b++) out.hbar[(size_t)a * n + b] += hb * ci[a] * ci[b];
            }
        }
    }
    lap("B-tensor and h chains");
    // 4. the semicanonicalization: eps (all three blocks), the Fock cross blocks, and the rotation of the active orbitals carried by the densities
    std::vector<double> fbar((size_t)nmo * nmo, 0.0);
    for(int i = 0; i < nc; i++){
        for(int t = 0; t < na; t++) fbar[(size_t)i * nmo + nc + t] = A.fca[(size_t)i * na + t];
        for(int e = 0; e < nv; e++) fbar[(size_t)i * nmo + nc + na + e] = A.fcv[(size_t)i * nv + e];
    }
    for(int t = 0; t < na; t++) for(int e = 0; e < nv; e++) fbar[(size_t)(nc + t) * nmo + nc + na + e] = A.fav[(size_t)t * nv + e];
    std::vector<double> Md(a2, 0.0), grot(a2);
    leg_moment(na, 2, A.L1.data(), in.L1, Md.data());
    leg_moment(na, 4, A.GAMMA.data(), in.GAMMA, Md.data());
    leg_moment(na, 6, A.G3.data(), in.G3, Md.data());
    for(int i = 0; i < na; i++) for(int k = 0; k < na; k++) grot[(size_t)i * na + k] = Md[(size_t)i * na + k] - Md[(size_t)k * na + i];
    std::vector<double> gbar(a2, 0.0);
    pt_canon_vjp(n, naux, nc, na, nv, Csc, eps, R.L(), R.V.data(), R.h.data(), in.L1, out.Cbar.data(), A.ec.data(), A.ev.data(), 1e-8, &out, gbar.data(),
                 nullptr, A.ea.data(), grot.data(), &R.ri(), fbar.data(), true);
    lap("semicanonicalization adjoint");
    // 5. densities back to the CAS-SCF active basis (the adjoint of the forward rotation is the backward one); the Fock read the same 1-RDM
    std::vector<double> L1b(a2), L1o(a2), G2o(a4), G3o(a6);
    for(size_t k = 0; k < a2; k++) L1b[k] = A.L1[k] + gbar[k];
    rotate1(L1b.data(), Ua, na, L1o.data(), false);
    rotate2(A.GAMMA.data(), Ua, na, G2o.data(), false);
    rotate3(A.G3.data(), Ua, na, G3o.data(), false);
    // 6. the CI source of the reference roots whose densities were averaged
    const int nvec = (int)roots.size();
    std::vector<double> cv((size_t)nvec * R.Nd), cb((size_t)nvec * R.Nd, 0.0);
    for(int I = 0; I < nvec; I++){ std::vector<double> c = R.root_vector(roots[I]); std::copy(c.begin(), c.end(), cv.begin() + (size_t)I * R.Nd); }
    aldet_rdm_vjp(R, cv.data(), nvec, wts.data(), L1o.data(), G2o.data(), G3o.data(), cb.data());
    std::vector<double> cbar_all((size_t)R.ns_all * R.Nd, 0.0);
    for(int I = 0; I < nvec; I++) for(size_t d = 0; d < R.Nd; d++) cbar_all[(size_t)roots[I] * R.Nd + d] += cb[(size_t)I * R.Nd + d];
    lap("CI source (RDM adjoints)");
    // 7. orbital adjoint in the gauge of the CAS-SCF orbitals: C_sc = U^T C_0 blockwise
    CasObjects direct;
    direct.E = L.total;
    direct.D = out.hbar; direct.Lbar = std::move(out.Lbar); direct.Vbar = std::move(out.Vbar);
    direct.GC.assign((size_t)n * nmo, 0.0);
    {
        const int first[3] = {0, nc, nc + na}, dim[3] = {nc, na, nv};
        const double * U[3] = {Uc, Ua, Uv};
        for(int b = 0; b < 3; b++) for(int k = 0; k < dim[b]; k++) for(int i = 0; i < dim[b]; i++){
            const double u = U[b][(size_t)k * dim[b] + i]; if(u == 0.0) continue;
            for(int a = 0; a < n; a++) direct.GC[(size_t)a * nmo + first[b] + k] += u * out.Cbar[(size_t)(first[b] + i) * n + a];
        }
    }
    out = PTChainOut();
    if(rs) fprintf(out_stream, "DSRG-PT2 gradient (relax=once): E = % .12f (unrelaxed % .12f)\n", E_relax, L.total);
    else fprintf(out_stream, "DSRG-PT2 gradient: E = % .12f (reference % .12f, correlation % .12f)\n", L.total, L.Eref, L.total - L.Eref);
    direct.E = g_dsrg_grad.E;
    if(cas_lagrangian_objects_cbar(M, R, direct, cbar_all, std::vector<double>(), &g_dsrg_grad.total)){ g_dsrg_grad.status = 1; return 1; }
    lap("SA-CASSCF response (Z-vector)");
    g_dsrg_grad.done = true;
    return 0;
}

int dsrg_grad_contract(molecule * M){
    g_dsrg_grad.g.assign(3 * M->n_atoms, 0.0);
    deriv_nuclear_repulsion(M, g_dsrg_grad.g.data());
    return cas_geo_contract(M, *g_dsrg_grad.ref, g_dsrg_grad.total, g_dsrg_grad.g.data());
}

int dsrg_relax_gradient(molecule * M, const DsrgTapeIn & in, const double * Csc, const double * eps, const double * Uc, const double * Ua, const double * Uv,
                        const std::vector<int> & roots, const std::vector<double> & wts, aldet_data * dressed, const std::vector<double> & wd,
                        const std::vector<double> & E_dressed){
    if(!g_dsrg_grad.want || g_dsrg_grad.status) return g_dsrg_grad.status;
    const CasRef & R = *g_dsrg_grad.ref;
    const int na = R.na, ns = dressed->n_states[0]; const size_t a2 = (size_t)na * na, a4 = a2 * a2, Nd = R.Nd;
    // densities of the dressed roots (semicanonical active basis, the basis of the dressed operator)
    aldet_data X;
    X.get_dim(R.ci_template->n_act, R.ci_template->na, R.ci_template->nb, 1, R.ci_template->mult, 0);
    X.init_zero_vec(ns, 0);
    for(size_t d = 0; d < Nd * ns; d++) X.coef[0][d] = dressed->coef[0][d];
    X.transpose_ci(0);
    std::vector<double> gam((size_t)ns * a2, 0.0), Gam((size_t)ns * a4, 0.0);
    X.calc_DM_diag(gam.data(), 0);
    X.G_calc(Gam.data());
    // the operator on the tape, and the dressed energies it must reproduce: E_d = e0 + <gamma_d, h1> + 1/2 <Gamma_d, h2>
    DsrgRelaxOut ro;
    dsrg_tape_energy(in, nullptr, nullptr, nullptr, &ro);
    DsrgRelaxSeeds rs; std::vector<double> h1b(a2, 0.0), h2b(a4, 0.0);
    double E = 0, wsum = 0, dev = 0;
    for(int d = 0; d < ns; d++){
        double e = ro.e0;
        for(size_t k = 0; k < a2; k++) e += gam[(size_t)d * a2 + k] * ro.h1[k];
        for(size_t k = 0; k < a4; k++) e += 0.5 * Gam[(size_t)d * a4 + k] * ro.h2_chem[k];
        dev = std::max(dev, std::fabs(e - E_dressed[d]));
        if(wd[d] == 0.0) continue;
        E += wd[d] * E_dressed[d]; wsum += wd[d];
        for(size_t k = 0; k < a2; k++) h1b[k] += wd[d] * gam[(size_t)d * a2 + k];
        for(size_t k = 0; k < a4; k++) h2b[k] += 0.5 * wd[d] * Gam[(size_t)d * a4 + k];
    }
    fprintf(out_stream, "DSRG-PT2 gradient (relax=once): dressed operator on the tape reproduces the dressed energies to %.2e\n", dev);
    if(dev > 1e-8){ fprintf(out_stream, "ERROR: the dressed operator of the gradient does not reproduce the dressed energies\n"); g_dsrg_grad.status = 1; return 1; }
    if(wsum == 0.0){ fprintf(out_stream, "ERROR: no dressed root carries the target weight (unmatched root): no relaxed energy to differentiate\n"); g_dsrg_grad.status = 1; return 1; }
    rs.e0b = wsum; rs.h1b = h1b.data(); rs.h2b_chem = h2b.data();
    return dsrg_gradient(M, in, Csc, eps, Uc, Ua, Uv, roots, wts, &rs, E);
}

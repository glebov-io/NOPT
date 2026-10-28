#include <cstring>
#include <cstdio>
#include <cctype>
#include <vector>
#include <stdlib.h>
#include "common_vars.h"

using namespace std;

vector<const char *>rhf_kw{{"rhf"}};

vector<const char *>rhf_group_start{{"$rhf"}};

vector<const char *>rhf_group_end{{"$rhfend"},
                                  {"$end"}};

vector<const char *>rhf_huckel{{"huckel"}};
                         
vector<const char *>rhf_h_core{{"h_core"}};
                  
vector<const char *>rhf_read{{"read"},
                             {"moread"},
                             {"readmo"},
                             {"read_orb"},
                             {"read_orbs"}};

vector<const char *>guess_kw{{"guess"}};

vector<const char *>guess_huckel_kw{{"huckel"}};

vector<const char *>guess_sad_kw{{"sad"}};
                         
vector<const char *>max_it_kw{{"maxit"},
                              {"max_iter"},
                              {"max_iters"}};

vector<const char *>e_conv_kw{{"de"},
                              {"deltae"},
                              {"delta_e"}};
                        
vector<const char *>g_conv_kw{{"grad"},
                              {"orb_grad"}};
                              
vector<const char *>s_conv_kw{{"step"}};
                              
vector<const char *>x_max_kw{{"x_max"}};

                              
vector<const char *>r_conv_kw{{"dr"},
                              {"deltar"},
                              {"delta_r"}};
                        
vector<const char *>cas_kw{{"cas"},
                           {"casscf"}};

vector<const char *>cas_SA_kw{{"sa"},
                              {"averaged"}};
                           
vector<const char *>cas_DA_kw{{"da"},
                              {"density_averaged"}};
                           
vector<const char *>cas_SM_kw{{"sm"},
                              {"separete"},
                              {"separate"}};
                           
                           
vector<const char *>cas_group_start{{"$cas"},
                                    {"$casscf"}};

vector<const char *>cas_group_end{{"cas_end"},
                                  {"casend"},
                                  {"casscf_end"},
                                  {"casscfend"},
                                  {"$end"}};
                        
vector<const char *>cisolver_kw{{"cisolver"}};

vector<const char *>cisolver_aldet_kw{{"aldet"}};

vector<const char *>cisolver_dmrg_kw{{"dmrg"}};

vector<const char *>converger_kw{{"converger"}};

vector<const char *>converger_soscf_kw{{"soscf"}};

vector<const char *>converger_sxpt_kw{{"sxpt"}};

vector<const char *>cas_lbfgs_kw{{"lbfgs"}};

vector<const char *>num_state_kw{{"n_s"},
                                 {"numstate"},
                                 {"num_state"}};

vector<const char *>num_frozen_orb_kw{{"n_f"},
                                      {"num_frozen_orb"},
                                      {"num_frozen_orbitals"}};

vector<const char *>method_kw{{"method"}};

vector<const char *>nat_orb_kw{{"no"},
			       {"nat_orb"},
                               {"natural_orbitals"}};

vector<const char *>cis_kw{{"cis"}};

vector<const char *>cis_group_start{{"$cis"}};

vector<const char *>cis_group_end{{"cis_end"},
                                  {"cisend"},
                                  {"$end"}};

vector<const char *>mp2_kw{{"mp2"}};

vector<const char *>mp2_group_start{{"$mp2"}};

vector<const char *>mp2_group_end{{"mp2_end"},
                                  {"mp2end"},
                                  {"$end"}};

vector<const char *>w_state_kw{{"wstate"},
                               {"w_state"}};
                               
vector<const char *>w_state_eq_kw{{"all_1"},
                                  {"eq"},
                                  {"equal"}};
                                  
vector<const char *>w_state_irrep_kw{{"rep"},
                                     {"ir.rep"},
                                     {"irrep"}};
                                  
vector<const char *>cas_track_kw{{"track"},
                                 {"tracking"}};

vector<const char *>cas_rotate_orbs_kw{{"rotate"}};

vector<const char *>w_state_group_start{{"$state"},
                                        {"$wstate"},
                                        {"$w_state"}};

vector<const char *>w_state_group_end{{"$stateend"},
                                      {"$wstateend"},
                                      {"$w_state_end"},
                                      {"$end"}};

vector<const char *>w_linear_L_kw{{"lambda"}};

vector<const char *>w_linear_sign_kw{{"sign"}};

                                 
vector<const char *>mult_kw{{"mult"}};

vector<const char *>print_number_kw{{"p_n"},
                                    {"p_num"},
                                    {"print_number"}};

vector<const char *>reorder_kw{{"reorder"},
                               {"norder"}};
                         
vector<const char *>orbitals_kw{{"orbitals"},
                                {"act_orb"},
                                {"active"}};
                         
                              
vector<const char *>dav_group_start{{"$dav"}};

                              
// vector<const char *>max_dav_it_kw{{"dav_it"},
//                                   {"DAV_IT"},
//                                   {"davit"},
//                                   {"DAVIT"}};

vector<const char *>dav_se_kw{{"s_ev"},
                              {"s_eval"}};

vector<const char *>dav_n_bf_kw{{"n_bf"}};

vector<const char *>dav_sparsed_HC_kw{{"sparsed_hc"}};
                          
vector<const char *>dav_group_end{{"dav_end"},
                                  {"davend"},
                                  {"$end"}};

vector<const char *>dmrg_group_start{{"$dmrg"}};

vector<const char *>dmrg_group_end{{"$dmrgend"},
                                   {"$dmrg_end"},
                                   {"$end"}};

vector<const char *>dmrg_m_kw{{"m"}};

vector<const char *>dmrg_sweeps_kw{{"sweeps"}};

vector<const char *>dmrg_sweep_tol_kw{{"sweep_tol"}};

vector<const char *>dmrg_hf_occ_kw{{"hf_occ"}};

vector<const char *>dmrg_hf_occ_integral_kw{{"integral"}};

vector<const char *>dmrg_schedule_kw{{"schedule"}};

vector<const char *>dmrg_schedule_default_kw{{"default"}};

vector<const char *>dmrg_save_dir_kw{{"save_dir"}};

vector<const char *>dmrg_memory_kw{{"memory"}};

vector<const char *>dmrg_rdm_passes_kw{{"rdm_passes"}};

vector<const char *>dmrg_main_stack_kw{{"main_stack"}};

vector<const char *>dmrg_localize_kw{{"localize"}};
vector<const char *>dmrg_localize_off_kw{{"off"}};
vector<const char *>dmrg_localize_pm_kw{{"pm"}};
vector<const char *>dmrg_localize_boys_kw{{"boys"}};
vector<const char *>dmrg_dump_loc_kw{{"dump_loc_orbs"}};
vector<const char *>dmrg_loc_order_kw{{"loc_order"}};
vector<const char *>dmrg_loc_order_fiedler_kw{{"fiedler"}};
vector<const char *>dmrg_loc_order_gaopt_kw{{"gaopt"}};
vector<const char *>dmrg_loc_order_none_kw{{"none"}};
vector<const char *>dmrg_warm_start_kw{{"warm_start"}};
vector<const char *>dmrg_warm_on_kw{{"on"}};
vector<const char *>dmrg_warm_off_kw{{"off"}};
vector<const char *>dmrg_warm_sweeps_kw{{"warm_sweeps"}};
vector<const char *>dmrg_warm_noise_scale_kw{{"warm_noise_scale"}};
vector<const char *>dmrg_rot_m_kw{{"rot_m"}};
vector<const char *>dmrg_rot_steps_kw{{"rot_steps"}};
vector<const char *>dmrg_warm_start_after_kw{{"warm_start_after"}};
vector<const char *>dmrg_warm_rotate_kw{{"warm_rotate"}};
vector<const char *>dmrg_print_dets_kw{{"print_dets"}};
vector<const char *>dmrg_det_rot_m_kw{{"det_rot_m"}};
vector<const char *>dmrg_det_rot_steps_kw{{"det_rot_steps"}};
vector<const char *>dmrg_extract_m_kw{{"extract_m"}};
vector<const char *>dmrg_extract_cutoff_kw{{"extract_cutoff"}};
vector<const char *>dmrg_h2caa_m_kw{{"h2caa_m"}};
vector<const char *>dmrg_low_m_opt_kw{{"low_m_opt"}};

vector<const char *>avas_group_start{{"$avas"}};

vector<const char *>avas_group_end{{"$avasend"},
                                   {"$avas_end"},
                                   {"$end"}};

vector<const char *>avas_atoms_kw{{"atoms"}};
vector<const char *>avas_shells_kw{{"shells"}};
vector<const char *>avas_ref_basis_kw{{"ref_basis"}};

vector<const char *>MO_group_start{{"$vec1"},
                                   {"$vec"},
                                   {"$vec_a_1"},
                                   {"$vec_a"},
                                   {"_mo"},
                                   {"[mo]"}};
                                   
vector<const char *>MO_text_group_start{{"$vec1"},
                                        {"$vec"},
                                        {"$vec_a_1"},
                                        {"$vec_a"},
                                        {"[mo]"}};


vector<const char *>MO_group_b_start{{"$vec_b_1"},
                                     {"$mo_b"},
                                     {"$vec_b"}};

vector<const char *>MO_group_end{{"$vecend1"},
                                 {"$vecend"},
                                 {"$end"}};

vector<const char *>basis_kw{{"_basis"}};
                       
vector<const char *>basis_end_kw{{"basisend"},
                                 {"basis_end"}};

vector<const char *>ri_basis_kw{{"_ribasis"},
                                {"_ri_basis"},
                                {"_auxbasis"},
                                {"_aux_basis"}};
                                
vector<const char *>auto_aux_kw{{"auto_aux"},
                                {"auto"},
                                {"ri_auto"}};

vector<const char *>ecp_kw{{"_ecp"}};
                                
vector<const char *>so_kw{{"so"},
                          {"spin_orbit"}};

vector<const char *>so_lib_kw{{"_so"},
                              {"_spin_orbit"}};

                          
vector<const char *>ecp_so_end_kw{{"ecp_end"},
                                  {"so_end"},
                                  {"ecpend"},
                                  {"soend"}};

vector<const char *>read_basis_kw{{"_fbasis"},
                                  {"_f_basis"}};

                          
                          
vector<const char *>geom_group_start{{"$geom1"},
                                     {"$geom"},
                                     {"$geometry"},
                                     {"_geom1"},
                                     {"_geom"},
                                     {"_geometry"},
                                     {"_xyz"},
                                     {"_zmat"},
                                     {"_z-mat"},
                                     {"[atoms]"}};
                               
vector<const char *>z_mat_kw{{"zmat"},
                             {"z-mat"}};
                             
vector<const char *>act_space_group_start{{"$act_space"},
                                          {"$active_space"},
                                          {"$states1"},
                                          {"$states"}};
                             
vector<const char *>act_space_group_end{{"$act_space_end"},
                                        {"$active_space_end"},
                                        {"$statesend1"},
                                        {"$statesend"},
                                        {"$end"}};

vector<const char *>geom_dim_unit{{"_unit"},
                                  {"_units"}};

                               
vector<const char *>geom_dim_au{{"au"},
                                {"bohr"}};


vector<const char *>geom_dim_nm{{"nm"},
                                {"nanometer"}};

vector<const char *>geom_dim_ang{{"ang"},
                                 {"angstrom"}};
                           
vector<const char *>wf_type{{"$wf_type"}};

vector<const char *>wf_type_end{{"$wf_type_end"}};

vector<const char *>uhf{{"uhf"}};

vector<const char *>wide{{"wide"}};

vector<const char *>energy_group_start{{"$energy1"},
                                       {"$energy"},
                                       {"$energy_a_1"},
                                       {"$energy_a"}};

vector<const char *>energy_group_b_start{{"$energy_b_1"},
                                         {"$energy_b"}};

vector<const char *>energy_group_end{{"$energy_end"},
                                     {"$energyend"},
                                     {"$end"}};
                               
                               
vector<const char *>par_group{{"$par"}};
                    
vector<const char *>mol_group{{"$mol"}};
                        
vector<const char *>par_group_end{{"$par_end"},
                                  {"$parend"},
                                  {"$end"}};
                            
vector<const char *>mol_group_end{{"$molend"},
                                  {"$molend1"},
                                  {"$end"},
                                  {"$end1"}};

vector<const char *>num_threads_kw{{"num_threads"},
                                   {"num_thread"},
                                   {"numthreads"},
                                   {"numthread"},
                                   {"n_threads"},
                                   {"n_thread"},
                                   {"nthreads"},
                                   {"nthread"}};
                       
vector<const char *>n_mol_kw{{"n_mol"},
                             {"num_mol"},
                             {"nmol"},
                             {"nummol"},
                             {"n_mols"},
                             {"num_mols"},
                             {"nmols"},
                             {"nummols"}};
                       
vector<const char *>doci_dec_kw{{"dec"}};
                       
vector<const char *>out_folder_kw{{"outfolder"},
                                  {"out_folder"}};
                            
vector<const char *>n_frags_kw{{"n_frag"},
                               {"num_frag"},
                               {"nfrag"},
                               {"numfrag"},
                               {"n_frags"},
                               {"num_frags"},
                               {"nfrags"},
                               {"numfrags"}};
                         
vector<const char *>inp_kw{{"inps"},
                           {"inp"}};

vector<const char *>charge_kw{{"_charge"},
                              {"_charges"}};
                        
vector<const char *>name_kw{{"name"}};

vector<const char *>occ_group_start{{"$occ1"},
                                    {"$occ"}};
                              
vector<const char *>occ_group_end{{"$occend1"},
                                  {"$occend"},
                                  {"$end"}};
                      
vector<const char *>xmc_kw{{"xmc"},
                           {"xmcqdpt"}};
                     
vector<const char *>xmc_group_start{{"$xmc"},
                                    {"$xmcqdpt"}};

vector<const char *>xmc_group_end{{"$xmcend"},
                                  {"$xmcqdptend"},
                                  {"$end"}};
                            
vector<const char *>edshift_kw{{"edshift"}};

vector<const char *>avecoe_kw{{"avecoe"}};

vector<const char *>ifitd_kw{{"ifitd"}};

vector<const char *>xmc_d_only_kw{{"d_only"}};
                            
vector<const char *>xmc_n_fit_kw{{"n_fit"}};
                            
vector<const char *>xmc_n_fit_pol_kw{{"n_fit_pol"}};
                            
vector<const char *>cdas_kw{{"cdas"},
                            {"cqd"},
                            {"cqdas"}};
                     
vector<const char *>cdas_group_start{{"$cdas"},
                                     {"$cqd"},
                                     {"$cqdas"}};

vector<const char *>cdas_group_end{{"$cdas_end"},
                                   {"$cqd_end"},
                                   {"$cqdas_end"},
                                   {"$end"}};
                            
vector<const char *>cdas_ipea_kw{{"ipea"}};
                          
vector<const char *>cdas_mppt_kw{{"mppt"}};
                          
vector<const char *>cdas_homo_kw{{"homo"}};
                          
vector<const char *>cdas_actual_kw{{"actual"}};
                            
vector<const char *>cdas_sing_en_kw{{"energy"}};
                            
vector<const char *>cdas_mult_en_kw{{"energies"}};
                             
vector<const char *>cdas_orb_en_kw{{"use_orb_for_energy"}};
                            
vector<const char *>cdas_fit_en_kw{{"use_firefly_fit_energy"}};
                                   
vector<const char *>cdas_rot_orbs{{"rotate"}};

vector<const char *>cdas_mode_kw{{"cdas_mode"}};

vector<const char *>cdas_skip_gno_scalar_kw{{"skip_gno_scalar"}};

vector<const char *>cdas_m_delta_kw{{"m_delta"}};

vector<const char *>cdas_m_cdas_kw{{"m_cdas"}};

vector<const char *>dsrg_kw{{"dsrg"}};

vector<const char *>dsrg_group_start{{"$dsrg"}};

vector<const char *>dsrg_group_end{{"$dsrgend"},
                                   {"$dsrg_end"},
                                   {"$end"}};

vector<const char *>dsrg_s_kw{{"s"}};

vector<const char *>dsrg_ccvv_source_kw{{"ccvv_source"}};

vector<const char *>dsrg_ccvv_normal_kw{{"normal"}};

vector<const char *>dsrg_ccvv_zero_kw{{"zero"}};

vector<const char *>dsrg_root_kw{{"root"}};

vector<const char *>dsrg_print_kw{{"print"}};

vector<const char *>dsrg_sa_kw{{"sa"}};

vector<const char *>dsrg_relax_kw{{"relax"}};

vector<const char *>dsrg_relax_none_kw{{"none"}};

vector<const char *>dsrg_relax_once_kw{{"once"}};

vector<const char *>pt1_dipole_kw{{"pt1_dipole"}};

vector<const char *>d5_kw{{"d5"}};
                            
vector<const char *>ri_kw{{"ri"}};

vector<const char *>n_st_kw{{"n_st"},
                            {"num_st"},
                            {"nst"},
                            {"numst"},
                            {"n_sts"},
                            {"num_sts"},
                            {"nsts"},
                            {"numsts"}};
                        
vector<const char *>n_act_kw{{"n_act"},
                             {"nact"},
                             {"n_val"},
                             {"nval"}};
                        
vector<const char *>n_alp_kw{{"n_alp"},
                             {"nalp"}};


vector<const char *>n_bet_kw{{"n_bet"},
                             {"nbet"}};

vector<const char *>backup_group_start{{"$backup"}};

vector<const char *>backup_group_end{{"$backupend"},
                                     {"$end"}};

vector<const char *>h1_backup_kw{{"h1"}};
                       
vector<const char *>h2_backup_kw{{"h2"}};
                        
vector<const char *>ri_backup_kw{{"ri"}};


                        
vector<const char *>prefix_backup_kw{{"prefix"}};

vector<const char *>n_bfgs_vec_kw{{"n_bfgs_vec"},
                                  {"n_bfgs"},
                                  {"num_bfgs_vec"},
                                  {"num_bfgs"}};

vector<const char *>prop_group{{"$prop"}};
                         
vector<const char *>prop_group_end{{"$propend"},
                                   {"$prop_end"},
                                   {"$end"}};
                             
vector<const char *>print_dipole_kw{{"dipole"}};
                         
// vector<const char *>first_order_property_kw{{"dipole1"},
//                                             {"DIPOLE1"}};
//                                             {"CDAS_FO"}};
//                                             {"DIPOLE"}};

vector<const char *>print_dispersion_kw{{"disp"}};

vector<const char *>print_quadrupole_kw{{"quad"}};

vector<const char *>print_mulliken_kw{{"mulliken"}};
                                
vector<const char *>symm_group_start{{"$symm"},
                                     {"$sym"}};
                                
vector<const char *>symm_group_end{{"$symmend"},
                                   {"$symm_end"},
                                   {"$end"}};
                             
vector<const char *>symm_point_group_kw{{"group"}};

vector<const char *>linear_kw{{"lin"},
                              {"linear"}};


vector<const char *>include_kw{{"#include"}};



vector<const char *>lib_kw{{"lib"},
                           {"library"}};
                              
                            
int failed_find_symbol(char * inp, vector<const char *> keywords, char s){
    
    fprintf(out_stream,"ERROR: failed reading one of the keywords:\n");
    for(const auto&k:keywords)
    fprintf(out_stream,"       - %s\n",k);
    fprintf(out_stream,"       Could not find \"%c\" in line:\n       %s\n",s,inp);
    exit(0);
    
    return 1;
}

int is_splitter(char a){
    
    if(a==' ')return 1;
    if(a==',')return 1;
    if(a==';')return 1;
    if(a=='\n')return 1;
    if(a=='\r')return 1;
    if(a=='=')return 1;

//     if(a==" ")return 1;
    return 0;
}


int valid_pointer(char * a, char *b){
    if(a==nullptr) return 0;
    if(a==b) return 1;
    return is_splitter(a[-1]);
}

// case-insensitive strstr - avoids keyword duplication and related errors
char* ci_strstr(char* h, const char* n){
    for(;*h;h++){
        char *a=h; const char *b=n;
        while(*b&&tolower((unsigned char)*a)==tolower((unsigned char)*b)){a++;b++;}
        if(*b==0) return h;
    }
    return nullptr;
}

char *key_word_find(char * str, vector<const char *> keywords){

    std::vector<char *> vf;
    vf.resize(0);

    for(const auto&k:keywords){
        // Accept k only as a whole token: bounded by a splitter (or string
        // start) on the LEFT and by a splitter or end-of-string on the RIGHT.
        size_t kl = strlen(k);
        for(char *f=ci_strstr(str,k); f!=nullptr; f=ci_strstr(f+1,k)){
            char r = f[kl];
            if(valid_pointer(f,str)&&(r=='\0'||is_splitter(r))){
                if(vf.size()==0) vf.push_back(f);
                else if(f!=vf[0]) vf.push_back(f);
                break;
            }
        }
    }
    // fprintf(out_stream,"%d\n",vf.size());
    if(vf.size()==1)return vf[0];
    if(vf.size()>1){
        fprintf(out_stream,"ERROR: in line:\n");
        fprintf(out_stream,"       %s\n",str);
        fprintf(out_stream,"       found more than 1 keyword form list:\n");
        for(const auto&k:keywords)
            fprintf(out_stream,"       - %s\n",k);
        exit(1);
    }
    
    
    return nullptr;
    
}


int key_word_comp(char * str, vector<const char *> keywords){
    
    if(key_word_find(str, keywords)==nullptr) return 0;
    else                                      return 1;
    
//     for(const auto&k:keywords)
//         if(strstr(str,k)!=NULL){ 
//             return 1;
//         }
//     
//     return 0;
    
} 


int key_word_eq(char * str, vector<const char *> keywords){
    
    
    while(is_splitter(str[0]))str++;
    char *f;
    
    for(const auto&k:keywords){
        f=ci_strstr(str,k);
        if(f==str) return 1;
    }
    return 0;
    
}


int kw_to_i(char * inp, vector<const char *> keywords, int def_val){
    
    char* f;
    f = key_word_find(inp,keywords);
    if(f!=NULL){
        if(strstr(f,"=")==NULL)failed_find_symbol(inp,keywords,'=');
        return atoi(strstr(f,"=")+1);
    }
    return def_val;
}

float kw_to_f(char * inp, vector<const char *> keywords, float def_val){
    
    char* f;
    f = key_word_find(inp,keywords);
    if(f!=NULL){
        if(strstr(f,"=")==NULL)failed_find_symbol(inp,keywords,'=');            
        return atof(strstr(f,"=")+1);
    }
    return def_val;
}

int lenth(char* s){
//     fprintf(out_stream,"s = %s\n",s);
        
    int i=0;
    int r=0;
    while(is_splitter(s[i]))i++;
    while(is_splitter(s[i])==0){
//         fprintf(out_stream,"%c\n",s[i]);
        i++;
        r++;
    }
    return r;
}

char* kw_cpy(char * out, char * inp, int l){
    
    int i=0;
    while(is_splitter(inp[i]))i++;
    for(int j=0;j<l;i++,j++)
        out[j]=inp[i];
    out[l]='\0';
    
    return (inp+i);
}

int kw_to_kw(char * inp, vector<const char *> keywords, vector<const char *> keywords_2){
    
    char* f;
    char* s;
    f = key_word_find(inp,keywords);
//     if(nt==NULL)fprintf(out_stream,"not found\n");
    if(f!=NULL){
        if(strstr(f,"=")==NULL)failed_find_symbol(inp,keywords,'=');
        s=strstr(f,"=");while(is_splitter(s[0]))s++;
        if(key_word_eq(s,keywords_2))return 1;
    }
    return 0;
}

// kw_to_kw, but the value must end right after the matched keyword (splitter or end)
int kw_to_kw_exact(char * inp, vector<const char *> keywords, vector<const char *> keywords_2){
    
    char* f;
    char* s;
    f = key_word_find(inp,keywords);
    if(f!=NULL){
        if(strstr(f,"=")==NULL)failed_find_symbol(inp,keywords,'=');
        s=strstr(f,"=");while(is_splitter(s[0]))s++;
        for(const auto&k:keywords_2){
            if(ci_strstr(s,k)!=s) continue;
            char r = s[strlen(k)];
            if(r=='\0'||is_splitter(r)) return 1;
        }
    }
    return 0;
}


int kw_to_s(char ** name, char * inp, vector<const char *> keywords){
    
    char* f;
    char* s;
    f = key_word_find(inp,keywords);
    if(f!=NULL){
        if(strstr(f,"=")==NULL)failed_find_symbol(inp,keywords,'=');
        s=strstr(f,"=");while(is_splitter(s[0]))s++;
        int l = lenth(s);
        *name=new char[l+1];
        kw_cpy(*name,s,l);
//         getchar();
    }
    return 1;
}

int kw_to_s_w_def(char ** name, char * inp, vector<const char *> keywords, char * def_val){
    
    char* f;
    char* s;
    int r_val=1;
    f = key_word_find(inp,keywords);
    if(f!=NULL){
        if(strstr(f,"=")==NULL) {s=def_val;r_val=0;}
        else s=strstr(f,"=");while(is_splitter(s[0]))s++;
        
        int l = lenth(s);
        *name=new char[l+1];
        kw_cpy(*name,s,l);
//         getchar();
    }
    return r_val;
}

int kw_to_s_v(std::vector<char*>* name, char * inp, vector<const char *> keywords, int n){
    
    char* f;
    char* s;
    char* s_tmp;
    f = key_word_find(inp,keywords);
    if(f!=NULL){
        s=strstr(f,"=");while(is_splitter(s[0]))s++;
        for(int i=0;i<n;i++){
            int l = lenth(s);
            (*name)[i]=new char[l+1];
            s_tmp = kw_cpy((*name)[i],s,l);
            s=s_tmp;
        }

//         getchar();
    }
    return 1;
}

int kw_to_i_v(std::vector<int>* c, char * inp, vector<const char *> keywords, int n){
    
    std::vector<char*> c_char;
    c_char.resize(n);
    kw_to_s_v(&c_char,inp,keywords,n);
    for(int i=0;i<n;i++){
        (*c)[i]=atoi(c_char[i]);
        delete[]c_char[i];
    }
    
    return 1;
}

int kw_to_d_v(std::vector<double>* c, char * inp, vector<const char *> keywords, int n){
    
    c->resize(n);
    std::vector<char*> c_char;
    c_char.resize(n);
    kw_to_s_v(&c_char,inp,keywords,n);
    for(int i=0;i<n;i++){
        (*c)[i]=atof(c_char[i]);
        delete[]c_char[i];
    }
    
    return 1;
}


int kw_count(char * inp, vector<const char *> keywords, char end){
    char* f;
    char* s;
    char* s_tmp;
    int c=0;
    f = key_word_find(inp,keywords);
    if(f!=NULL){
        c=1;
        if(strchr(f,end)==NULL)failed_find_symbol(inp,keywords,end);
        if(strstr(f,"=")==NULL)failed_find_symbol(inp,keywords,'=');
        s=strstr(f,"=");while(is_splitter(s[0]))s++;
        int i=0;
        while(s[i]!=end){
            if(is_splitter(s[i]))c++;
            i++;
        }
    }
    
    return c;
}

# include "blas_link.h"


# include "MO2.h"
# include "molecule.h"
# include "molecule2.h"
# include "chem_data.h"
# include "matr.h"
# include "timer.h"
# include "from_hash.h"
# include "keywords.h"
# include "binary_files.h"
# include "etc.h"
#include "common_vars.h"
# include "RI.h"
# include "defaults.h"
# include "geom.h"
# include "ecp.h"
#include "basis_lib_read.h"
#include "sad_guess.h"
#include "inp_out.h"

// 
# ifdef LIBINT
// # include <vector>
// # include <libint2/atom.h>
// using libint2::Atom;
// using libint2::Shell;
# include "libint_link.h"
#endif


using namespace std;

int D_L[12]={0,0,
             0,1,
             0,2,
             1,1,
             1,2,
             2,2};

int F_L[30]={0,0,0,//0
             0,0,1,//1
             0,0,2,//2
             0,1,1,//3
             0,1,2,//4
             0,2,2,//5
             1,1,1,//6
             1,1,2,//7
             1,2,2,//8
             2,2,2};//9
             
int G_L[60]={0,0,0,0,
             0,0,0,1,
             0,0,0,2,
             0,0,1,1,
             0,0,1,2,
             0,0,2,2,
             0,1,1,1,
             0,1,1,2,
             0,1,2,2,
             0,2,2,2,
             1,1,1,1,
             1,1,1,2,
             1,1,2,2,
             1,2,2,2,
             2,2,2,2};
             
int H_L[105]={0,0,0,0,0,
              0,0,0,0,1,
              0,0,0,0,2,
              0,0,0,1,1,
              0,0,0,1,2,
              0,0,0,2,2,
              0,0,1,1,1,
              0,0,1,1,2,
              0,0,1,2,2,
              0,0,2,2,2,
              0,1,1,1,1,
              0,1,1,1,2,
              0,1,1,2,2,
              0,1,2,2,2,
              0,2,2,2,2,
              1,1,1,1,1,
              1,1,1,1,2,
              1,1,1,2,2,
              1,1,2,2,2,
              1,2,2,2,2,
              2,2,2,2,2};

int F_ind(int a, int b, int c){
    
    int l[3];
        l[0]=a;
        l[1]=b;
        l[2]=c;
    sort_int_array(l, 3);
    for(int i=0;i<10;i++)
        if(l[0]==F_L[3*i])if(l[1]==F_L[3*i+1])if(l[2]==F_L[3*i+2])return i;
        
    exit(1);
    return -1;
    
}

int G_ind(int a, int b, int c, int d){
    
    int l[4];
        l[0]=a;
        l[1]=b;
        l[2]=c;
        l[3]=d;
    sort_int_array(l, 4);
    for(int i=0;i<15;i++)
        if(l[0]==G_L[4*i])if(l[1]==G_L[4*i+1])if(l[2]==G_L[4*i+2])if(l[3]==G_L[4*i+3])return i;
        
    exit(1);
    return -1;
    
}

int H_ind(int a, int b, int c, int d, int e){
    
    int l[5];
        l[0]=a;
        l[1]=b;
        l[2]=c;
        l[3]=d;
        l[4]=e;
    sort_int_array(l, 5);
    for(int i=0;i<21;i++)
        if(l[0]==H_L[5*i])if(l[1]==H_L[5*i+1])if(l[2]==H_L[5*i+2])if(l[3]==H_L[5*i+3])if(l[4]==H_L[5*i+4])return i;
        
    exit(1);
    return -1;
    
}
             
             
molecule::molecule()
{
    n_atoms = 0;
    ecp=0;
    reorder=0;
    
    //// for finite field calculation of dipole moment, electric field strength default should be 0
    F_x=0; F_y=0; F_z=0; 
    
    atom_coord=NULL;
//     mcs_rot_matr = new double[9];
    nucl_charges_full=NULL;
    nucl_charges_calc=NULL;
    n_ecp_electrons  =NULL;
    atom_mass=NULL;
    atom_names=NULL;
    atom_is_ghost=NULL;
    
    MO_VEC                 =NULL;
    MO_VEC_B               =NULL;
    MO_VEC_R               =NULL;
    MO_BUF                 =NULL;
    orb_energy             =NULL;
    orb_energy_B           =NULL;
    n_act_el_alp           =NULL;
    n_act_el_bet           =NULL;
    n_act_orb              =NULL;
    n_states               =NULL;
    n_cor_orb_f            =NULL;
    
    
    basis_name         =NULL;
    ri_basis_name      =NULL;
    read_basis_name    =NULL;
    ecp_name           =NULL;
    so_name            =NULL;
    states_energies        =NULL;
    
    rep_num                =NULL;
    rep_num_backup         =NULL;
//     rep_AO_num             =NULL;
    
    n_CI=1;
    read_CI                =NULL;
    CI_file_name           =NULL;
    CI                     =NULL;
    CAS                    =NULL;
    
    nat_orbs_m             =NULL;
    NO_VEC                 =NULL;
    NO_VEC_B               =NULL;
    nat_orb_occ            =NULL;
    nat_orb_occ_B          =NULL;
    
    Lambda_AO              =NULL;
    

    source_name=new char[BUF_LINE_LENGTH];
    MO_name=NULL;
    
    S_AO   = NULL; 
    S_MO   = NULL; 
    S_M05  = NULL; 
    H_AO   = NULL; 
    Dx_AO  = NULL; 
    Dy_AO  = NULL; 
    Dz_AO  = NULL; 
    Qxx_AO = NULL;
    Qxy_AO = NULL;
    Qxz_AO = NULL;
    Qyy_AO = NULL;
    Qyz_AO = NULL;
    Qzz_AO = NULL;
    P_AO   = NULL;
    RxP_AO = NULL;
    BUF    = NULL; 
    V_D6   = NULL;
    T_D5   = NULL;
        
}

molecule::~molecule()
{ 
    
    if(n_atoms){
        if(atom_names!=NULL){
            for(int i=0;i<n_atoms;i++) delete [] atom_names[i];
            delete [] atom_names;
        }
        n_atoms = 0;
    }
//     delete[] mcs_rot_matr;
    
    if(CI_file_name     != NULL) {for(int i=0;i<n_frag;i++)if(CI_file_name[i]!=NULL)delete[] CI_file_name[i];delete [] CI_file_name;}
    
    if(MO_VEC_B    !=NULL)if(MO_VEC_B!=MO_VEC)delete [] MO_VEC_B;
    if(orb_energy_B!=NULL)if(orb_energy_B!=orb_energy)delete [] orb_energy_B;
    
    if(source_name        != NULL) delete [] source_name;
    if(MO_name            != NULL) delete [] MO_name;
    if(MO_VEC_R           != NULL) delete [] MO_VEC_R;
    if(MO_VEC             != NULL) delete [] MO_VEC;
    if(MO_BUF             != NULL) delete [] MO_BUF;
    if(n_act_el_alp       != NULL) delete [] n_act_el_alp;// rewrite!!!!
    if(n_act_el_bet       != NULL) delete [] n_act_el_bet;// rewrite!!!!
    if(n_act_orb          != NULL) delete [] n_act_orb;   // rewrite!!!!
    if(n_states           != NULL) delete [] n_states;
    
    if(read_CI            != NULL) delete [] read_CI;
    if(CI                 != NULL) delete [] CI;
    if(CAS                != NULL) delete [] CAS;
    if(        so_name!= NULL) if(        so_name!=   ecp_name)delete []         so_name;
    if(       ecp_name!= NULL) if(       ecp_name!= basis_name)delete []        ecp_name;
    if(read_basis_name!= NULL) if(read_basis_name!= basis_name)delete [] read_basis_name;
    if(basis_name     != NULL) delete [] basis_name;
    if(ri_basis_name  != NULL) delete [] ri_basis_name;
    if(states_energies    != NULL) delete [] states_energies;
    if(orb_energy         != NULL) delete [] orb_energy;
    if(n_cor_orb_f        != NULL) delete [] n_cor_orb_f;

    if(atom_coord       !=NULL) delete[] atom_coord;
    if(nucl_charges_full!=NULL) delete[] nucl_charges_full;
    if(nucl_charges_calc!=NULL) delete[] nucl_charges_calc;
    if(n_ecp_electrons  !=NULL) delete[] n_ecp_electrons  ;
    if(atom_mass        !=NULL) delete[] atom_mass;
    
    if(nat_orbs_m   != NULL)delete[] nat_orbs_m   ;
    if(NO_VEC       != NULL)delete[] NO_VEC       ;
    if(NO_VEC_B     != NULL)delete[] NO_VEC_B     ;
    if(nat_orb_occ  != NULL)delete[] nat_orb_occ  ;
    if(nat_orb_occ_B!= NULL)delete[] nat_orb_occ_B;
    
    if(rep_num       != NULL)delete[] rep_num       ;
    if(rep_num_backup!= NULL)delete[] rep_num_backup;
//     if(rep_AO_num    != NULL)delete[] rep_AO_num    ;
    
    if(Lambda_AO     != NULL)delete[] Lambda_AO     ;
    
    if(S_AO   != NULL) delete[] S_AO   ; 
    if(S_MO   != NULL) delete[] S_MO   ; 
    if(S_M05  != NULL) delete[] S_M05  ; 
    if(H_AO   != NULL) delete[] H_AO   ; 
    if(Dx_AO  != NULL) delete[] Dx_AO  ; 
    if(Dy_AO  != NULL) delete[] Dy_AO  ; 
    if(Dz_AO  != NULL) delete[] Dz_AO  ; 
    if(Qxx_AO != NULL) delete[] Qxx_AO ;
    if(Qxy_AO != NULL) delete[] Qxy_AO ;
    if(Qxz_AO != NULL) delete[] Qxz_AO ;
    if(Qyy_AO != NULL) delete[] Qyy_AO ;
    if(Qyz_AO != NULL) delete[] Qyz_AO ;
    if(Qzz_AO != NULL) delete[] Qzz_AO ;
    if(P_AO   != NULL) delete[] P_AO   ;
    if(RxP_AO != NULL) delete[] RxP_AO ;
    if(V_D6   != NULL) delete[] V_D6   ;
    if(T_D5   != NULL) delete[] T_D5   ;
    if(BUF    != NULL) delete[] BUF    ; 
    
}
       
int molecule::geom_read(){
    
    recursive_file in;
    FILE * at;
    char * line = new char[BUF_LINE_LENGTH];
    char * at_name = new char[BUF_LINE_LENGTH];
    sprintf(at_name,"%s/atoms.dat\0",NOPT_LIB);
    
    
    int line_num=0;
    int start;
    
    
    in.r_open (source_name);
    at = fopen (    at_name,"r");
    
    for(int i=0;i<inp_mol_line;i++){in.r_gets(line,BUF_LINE_LENGTH),line_num++;}
    in.r_gets(line,BUF_LINE_LENGTH),line_num++;
    while((key_word_comp(line,charge_kw)==0)&&(!in.r_eof())){in.r_gets(line,BUF_LINE_LENGTH),line_num++;}
    if(key_word_comp(line,charge_kw)==0){
        fprintf(out_stream,"\n");
        fprintf(out_stream,"WARNING: molecule charge is not specified in %s\n",source_name);
        fprintf(out_stream,"         using default CHARGE=0\n\n");
        mol_charge=0;
    }
    else
        mol_charge=kw_to_i(line, charge_kw, 0);
    
    int molden_geom=0;
    
    in.go_to_begining();
    line_num=0;
    
    for(int i=0;i<inp_mol_line;i++){in.r_gets(line,BUF_LINE_LENGTH),line_num++;}
    in.r_smart_fgets(line,&line_num,source_name, "reading file");
    
    while((key_word_comp(line,geom_group_start)==0)&&(!in.r_eof()))in.r_smart_fgets(line,&line_num,source_name, "looking for molecular geometry (keywords: xyz, geom, geometry, z-mat)");
    if(key_word_comp(line,inp_kw)){
        char * new_name;
        kw_to_s(&new_name, line, inp_kw);
        fprintf(out_stream,"                      included   %s (geometry)\n",new_name);
        in.r_close();
        in.r_open (new_name);
        if(in.file[0]==NULL){
            fprintf(out_stream,"Couldn't open '%s'\n",new_name);
            exit(0);
        }
        if(strstr(new_name,".molden")!=NULL)molden_geom=1;
        delete[] new_name;
    }
    
    if(molden_geom){
        printf("MOLDEN geometry is not yet supported\n");
        exit(0);
    }
    
    int format =0;//xyz
//     printf("%s\n", line);
//     getchar();
    if(key_word_comp(line,z_mat_kw))format=1;//z-matrix
    
    
    in.r_smart_fgets(line,&line_num,source_name,"reading number of atoms");
    
    n_atoms=atoi(line);
    n_el_full = 0;
    n_el_calc = 0;
    if(n_atoms<1){
        fprintf(out_stream,"ERROR: line %d in %s\n", line_num, source_name);
        fprintf(out_stream,"       number of atoms is illegal n=%d\n",n_atoms);
        exit(1);
    }
    
    atom_names        = new char* [n_atoms];
    atom_coord        = new double[n_atoms*3];
    atom_mass         = new double[n_atoms];
    atom_is_ghost     = new int   [n_atoms];
    nucl_charges_full = new double[n_atoms];
    nucl_charges_calc = new double[n_atoms];
    n_ecp_electrons   = new int   [n_atoms];
    set_zero_matr_int(n_ecp_electrons,n_atoms);
    z.init(n_atoms);
    
    in.r_smart_fgets(line,&line_num,source_name,"reading xyz comment");
    for(int i=0;i<n_atoms;i++){
        
        in.r_smart_fgets(line,&line_num,source_name,"reading atom");
        atom_names[i] = new char[3];
        
        for(start=0;start<strlen(line);start++)if(isalpha(line[start])){
//             fprintf(out_stream,"!%c! %d\n",line[start],start);
            goto read_atom;
        }
        
        fprintf(out_stream,"ERROR: could not find atom name in %s\n",source_name);
        fprintf(out_stream,"       line %d of length %d\n",line_num, strlen(line));
        exit(1);
        
        read_atom:;
        atom_names[i][0]=line[start  ];
        atom_names[i][1]=line[start+1];
        atom_names[i][2]='\0';
        if(format==0){
            std::stringstream in(line+start+2);
            in>>atom_coord[3*i+0];
            in>>atom_coord[3*i+1];
            in>>atom_coord[3*i+2];
            if(!(in>>nucl_charges_calc[i]))nucl_charges_calc[i]=0;
        }
        else
            z.read_line(line+start+2, i);
      
        nucl_charges_full[i] =charge(atom_names[i], at);
        nucl_charges_calc[i]+=charge(atom_names[i], at);
        atom_mass        [i] =mass  (atom_names[i], at);
        atom_is_ghost    [i] =ghost (atom_names[i], at);
        // printf("%s is %s - %d\n",atom_names[i], (atom_is_ghost[i])?"ghost":"real ",atom_is_ghost[i]);
        // getchar();
        
        
        if(nucl_charges_full[i]<0){
            fprintf(out_stream,"ERROR: could not find atom %s in atoms.dat\n",atom_names[i]);
            fprintf(out_stream,"       line %d of %s\n",line_num,source_name);
            exit(1);
        }
        
        n_el_full+=nucl_charges_full[i];
        
    }
    
    if(format){
        in.r_smart_fgets(line,&line_num,source_name,"reading atom");
        if(strstr(line,"xyz-3")==NULL){
            printf("ERROR: could not read first atoms\n");
            printf("       xyz-3 not found\n");
            exit(0);
        }
        in.r_smart_fgets(line,&line_num,source_name,"reading atom");
        z.read_xyz(line,0);
        in.r_smart_fgets(line,&line_num,source_name,"reading atom");
        z.read_xyz(line,1);
        in.r_smart_fgets(line,&line_num,source_name,"reading atom");
        z.read_xyz(line,2);
    }
    fprintf(out_stream,"\nWARNING: read_cl is blocked in the 3.0 version\n\n");
    //read_cl(&in);
    
    n_el_full-=mol_charge;
    n_el_calc=n_el_full;
    
    if(molden_geom==0){
        in.r_close();
        in.r_open (source_name);
    }
    else
        in.go_to_begining();
    
    line_num=0;
    in.r_gets(line,BUF_LINE_LENGTH),line_num++;
    while((key_word_comp(line,geom_dim_unit)==0)&&(!in.r_eof())){
//         printf("%d    %s\n",line_num,line);
        in.r_gets(line,BUF_LINE_LENGTH),line_num++;
    }
    
    if(key_word_comp(line,geom_dim_unit)==0){
        fprintf(out_stream,"                      units    = ang\n");
        au_coef=1.889725989;
    }
    else{
        if(key_word_comp(line,geom_dim_au )) {au_coef=1.0        ;fprintf(out_stream,"                      units    = bohr\n");}
        if(key_word_comp(line,geom_dim_nm )) {au_coef=18.89725989;fprintf(out_stream,"                      units    = nm\n");}
        if(key_word_comp(line,geom_dim_ang)) {au_coef=1.889725989;fprintf(out_stream,"                      units    = ang\n");}
    }
    
    for(int i=0;i<3*n_atoms;i++) atom_coord[i]=au_coef*atom_coord[i];
    z.scale_R(au_coef);
    dist_scale(au_coef);
    
    
    
    if(format){
        coord_sync(z.p);
        z.calc_xyz(atom_coord);
//         move_to_mcs(1,1);
    }
    else      z.calc_Z  (atom_coord);
    
    
    
//     z.print();
//     GAMESS_geom_print("tmp.out");
//     exit(1);
    
    
//     fprintf(out_stream,"n_e=%d\nc=%d\n",n_el,mol_charge);
//     getchar();
    
    in.r_close();
    fclose(at);
    delete[] line   ;
    delete[] at_name;
    
    
    return 0;
}

int molecule::geom_print()
{
  
    fprintf(out_stream,"Molecule structure (a.u.):\n");
    
    fprintf(out_stream," ________________________________________________________________________________________\n");
    fprintf(out_stream,"| Atom |    C    |     X     |     Y     |     Z     |   n_ecp   |     q     |    type   |\n");
    fprintf(out_stream,"| _____|_________|___________|___________|___________|___________|___________|___________|\n");
    
    for(int i=0;i<n_atoms;i++){
        fprintf(out_stream,"|  %s  ",atom_names[i]);
        
        if(fabs(nucl_charges_full[i])>1e-5)fprintf(out_stream,"| %5.0f   ",nucl_charges_full[i]);
        else                               fprintf(out_stream,"|         ");
        
        fprintf(out_stream,"|% 10.3f |% 10.3f |% 10.3f ",
                atom_coord[3*i  ],
                atom_coord[3*i+1],
                atom_coord[3*i+2]);
        if(n_ecp_electrons[i])fprintf(out_stream,"|   %5d   ",n_ecp_electrons[i]);
        else                  fprintf(out_stream,"|           ");
        double q = nucl_charges_calc[i] - nucl_charges_full[i]+n_ecp_electrons[i];
        if(fabs(q)>1e-5)fprintf(out_stream,"|% 10.5f |",q);
        else            fprintf(out_stream,"|           |");
        
        fprintf(out_stream,"    %s  |\n",(atom_is_ghost[i])?"ghost":"real ");
    }
    fprintf(out_stream,"| _____|_________|___________|___________|___________|___________|___________|___________|\n");
    
    if(IS_SYM)fprintf(out_stream,"\n%s symmetry point group is used\n",S.group);
    fprintf(out_stream,"\nMolecule contains %d electrons\n\n",n_el_calc);
    
    fprintf(out_stream,"C - nuclear charge (a.u.)\n");
    fprintf(out_stream,"q - additional (partial) charge (a.u.)\n");
    fprintf(out_stream,"n_ecp - number of electrons frozen by ECP\n");
    fprintf(out_stream,"X,Y,Z - cartesian coordinates (Bohr)\n");
    if((fabs(F_x)+fabs(F_y)+fabs(F_z))>=1e-10){
        fprintf(out_stream,"Warning: H was modified for this molecule\n");
        fprintf(out_stream,"Electric field (a.u.) is nonzero, F_x = %10.6f , F_y = %10.6f , F_z = %10.6f\n",F_x,F_y,F_z);
    }
    fprintf(out_stream,"\n\n\n");
    
    return 0;
}

int molecule::basis_name_read(){
    
    recursive_file in;
    char line[BUF_LINE_LENGTH];
    
    in.r_open(source_name);
    int line_num=0;
    if (in.file[0]==NULL) {fprintf(out_stream,"ERROR: Unable to open source file%s\n",source_name);exit(1);}
    
    //reading basis name
    
    for(int i=0;i<inp_mol_line;i++){in.r_gets(line,BUF_LINE_LENGTH);line_num++;}
    in.r_gets(line,BUF_LINE_LENGTH);line_num++;
    while((key_word_comp(line,basis_kw)==0)&&(!in.r_eof())) {in.r_gets(line,BUF_LINE_LENGTH);line_num++;}
    
    if (key_word_comp(line,basis_kw)==0)
    {
        in.r_close();
        fprintf(out_stream,"ERROR: Didn't find basis in %s\n",source_name);
        exit(1);
    }
    
    if(basis_name  != NULL) delete [] basis_name;// kw_to_s will allocate basis_name
    if(kw_to_s_w_def(&basis_name, line, basis_kw, source_name)==0){
            basis_in_lib=0;
            basis_start_line=line_num;
    }
    else{
            basis_in_lib=1;
            basis_start_line=0;
    }
    
    basis_format=0;//gbs format
    if(strstr(basis_name,".exp")){
        basis_format=1;//exp format
        if(key_word_comp(line,lib_kw)) basis_in_lib=1;
        else basis_in_lib=0;
    }
    else
        if (strstr(line,".exp"))basis_format=1;
        
    if(strstr(basis_name,".gbs")){
        basis_format=0;//gbs format
        if(key_word_comp(line,lib_kw)) basis_in_lib=1;
        else basis_in_lib=0;
    }
    
            
    
    if(RI){
        
        in.go_to_begining();line_num=0;
        for(int i=0;i<inp_mol_line;i++){in.r_gets(line,BUF_LINE_LENGTH);line_num++;}
        in.r_gets(line,BUF_LINE_LENGTH);line_num++;
        while((key_word_comp(line,ri_basis_kw)==0)&&(!in.r_eof())) {in.r_gets(line,BUF_LINE_LENGTH);line_num++;}
        
        if (key_word_comp(line,ri_basis_kw)==0){
            in.r_close();
            fprintf(out_stream,"ERROR: Didn't find RI basis in %s\n",source_name);
            exit(1);
        }
        if (key_word_comp(line,auto_aux_kw)){
            in.r_close();
            printf("AUTO_AUX is not yet realized\n");
            exit(1);
        }
        else{
            if(kw_to_s_w_def(&ri_basis_name, line, ri_basis_kw, source_name)==0){
                    ri_basis_in_lib=0;
                    ri_basis_start_line=line_num;
            }
            else{
                    ri_basis_in_lib=1;
                    ri_basis_start_line=0;
            }
            ri_basis_format=0;//gbs format
            if(strstr(ri_basis_name,".exp")){
                ri_basis_format=1;//exp format
                if(key_word_comp(line,lib_kw)) ri_basis_in_lib=1;
                else ri_basis_in_lib=0;
            }
            else
                if (strstr(line,".exp"))ri_basis_format=1;
                
            if(strstr(ri_basis_name,".gbs")){
                ri_basis_format=0;//gbs format
                if(key_word_comp(line,lib_kw)) ri_basis_in_lib=1;
                else ri_basis_in_lib=0;
            }
        }
            
    
    }
    
    in.go_to_begining();line_num=0;
    for(int i=0;i<inp_mol_line;i++){in.r_gets(line,BUF_LINE_LENGTH);line_num++;}
    in.r_gets(line,BUF_LINE_LENGTH);line_num++;
    while((key_word_comp(line,ecp_kw)==0)&&(!in.r_eof())){in.r_gets(line,BUF_LINE_LENGTH);line_num++;}
    
    if (key_word_comp(line,ecp_kw)==0){
        ecp_name   = basis_name;
        ecp_in_lib     = basis_in_lib;
        ecp_format     = basis_format;
        ecp_start_line = basis_start_line;
    }
    else{
        if(kw_to_s_w_def(&ecp_name, line, ecp_kw, source_name)==0){
            ecp_in_lib=0;
            ecp_start_line=line_num;
        }
        else{
                ecp_in_lib=1;
                ecp_start_line=0;
        }
        
        ecp_in_lib=kw_to_s_w_def(&ecp_name, line, ecp_kw, source_name);
        ecp_format=0;//gbs format
        if(strstr(ecp_name,".exp")){
            ecp_format=1;//exp format
            if(key_word_comp(line,lib_kw)) ecp_in_lib=1;
            else ecp_in_lib=0;
        }
        else
            if (strstr(line,".exp"))ecp_format=1;
        
        if(strstr(ecp_name,".grp")){
            ecp_format=2;//exp format
            if(key_word_comp(line,lib_kw)) ecp_in_lib=1;
            else ecp_in_lib=0;
        }
        else
            if (strstr(line,".grp"))ecp_format=2;
        
            
            
        if(strstr(ecp_name,".gbs")){
            ecp_format=0;//gbs format
            if(key_word_comp(line,lib_kw)) ecp_in_lib=1;
            else ecp_in_lib=0;
        }
    }
    
    if(SO){
        in.go_to_begining();line_num=0;
        for(int i=0;i<inp_mol_line;i++){in.r_gets(line,BUF_LINE_LENGTH);line_num++;}
        in.r_gets(line,BUF_LINE_LENGTH);line_num++;
        while((key_word_comp(line,so_lib_kw)==0)&&(!in.r_eof())) {in.r_gets(line,BUF_LINE_LENGTH);line_num++;}
        
        if (key_word_comp(line,so_lib_kw)==0){
            so_name   = ecp_name;
            so_in_lib     = ecp_in_lib;
            so_format     = ecp_format;
            so_start_line = ecp_start_line;
        }
        else{
            if(kw_to_s_w_def(&so_name, line, so_lib_kw, source_name)==0){
                so_in_lib=0;
                so_start_line=line_num;
            }
            else{
                so_in_lib=1;
                so_start_line=0;
            }
            
            so_format=0;//gbs format
            if(strstr(so_name,".exp")){
                so_format=1;//exp format
                if(key_word_comp(line,lib_kw)) so_in_lib=1;
                else so_in_lib=0;
            }
            else
                if (strstr(line,".exp"))so_format=1;
            
            if(strstr(so_name,".grp")){
                so_format=2;//grpp format
                if(key_word_comp(line,lib_kw)) so_in_lib=1;
                else so_in_lib=0;
            }
            else
                if (strstr(line,".grp"))so_format=2;
        
            if(strstr(so_name,".gbs")){
                so_format=0;//gbs format
                if(key_word_comp(line,lib_kw)) so_in_lib=1;
                else so_in_lib=0;
            }
        }
    }
    
    in.r_close();
    
    return 0;
}

int molecule::basis_print(int t){
    
    int n_s;
    int * c;
    Shell * b;
    std::vector<double> * lc;
    
    if(t==0){
        n_s = s.size();
        c   = shell_center.data();
        b   = s.data();
        fprintf(out_stream," Basis set information:\n\n");
        lc=lib_coef.data();
    }
    else{
        n_s = ri_s.size();
        c   = ri_shell_center.data();
        b   = ri_s.data();
        fprintf(out_stream," RI basis set information:\n\n");
        lc=lib_coef_ri.data();
    }
    
    fprintf(out_stream," ____________________________________________________________________________________\n");
    fprintf(out_stream,"|     Atom     |    L    |       alpha       |         c         |       c(lib)      |\n");
    fprintf(out_stream,"|______________|_________|___________________|___________________|___________________|\n");
    
    
    for(int i=0;i<n_atoms;i++){
        
        if(i)fprintf(out_stream,"|--------------|---------|-------------------|-------------------|-------------------|\n");
        fprintf(out_stream,"|   %s(%3d)    |         |                   |                   |                   |\n",atom_names[i],i);
        
        int new_sh=0;
        for(int sh_i=0;sh_i<n_s;sh_i++)
        if(c[sh_i]==i){
            if(new_sh)fprintf(out_stream,"|              |---------|-------------------|-------------------|-------------------|\n");
            fprintf(out_stream,"|              |   %3d   |% 18.6f |% 18.6f |% 18.6f |\n",
                               b[sh_i].contr[0].l,b[sh_i].alpha[0],b[sh_i].contr[0].coeff[0],lc[sh_i][0]);
            for(int k=1;k<b[sh_i].alpha.size();k++)
                fprintf(out_stream,"|              |         |% 18.6f |% 18.6f |% 18.6f |\n",
                                   b[sh_i].alpha[k],b[sh_i].contr[0].coeff[k],lc[sh_i][k]);
            new_sh=1;
        }
        
    }
    
    fprintf(out_stream,"|______________|_________|___________________|___________________|___________________|\n");
    fprintf(out_stream,"\n\n\n");
    
    
    return 0;
}

int molecule::PP_print(){
    
    
    fprintf(out_stream," Pseudopotential information:\n\n");
    fprintf(out_stream," ________________________________________________________________ \n");
    for(auto &pp: PP){
        pp.print(out_stream, atom_names[pp.atom],0);
        fprintf(out_stream,"|______________|_________|___________________|___________________|\n");
    }
    fprintf(out_stream,"\n\n\n");
    
    if(SO){
        fprintf(out_stream," Spin-orbital potential information:\n\n");
        
        fprintf(out_stream," ____________________________________________________________________________________ \n");
        for(auto &pp: SO_PP){
            pp.print(out_stream, atom_names[pp.atom],1);
            fprintf(out_stream,"|______________|_________|___________________|___________________|___________________|\n");
        }
        fprintf(out_stream,"\n\n\n");
    }
    
    return 0;
    
}



int molecule::active_space_read(int warning, int read_states, int ci_alloc){
    
    recursive_file in;
    char line[BUF_LINE_LENGTH];
    std::vector<int> act_orbs;
    reorder=0;
    int mult;
    int print_number = CAS_PRINT_NUMBER_DEFAULT;
    n_frag =1;
    
    n_act_el_alp=new int[1];
    n_act_el_bet=new int[1];
    n_act_orb   =new int[1];
    n_states    = new int[1];
    n_cor_orb_f = new int[1];
    CI          = new aldet_data[1];
    read_CI     = new int[1];
    CI_file_name= new char*[1];
    CI_file_name[0]=NULL;
    
    n_states[0]=1;
    n_act_el_alp[0]=0;
    n_act_el_bet[0]=0;
    n_act_orb[0]=0;
    n_cor_orb=0;
    read_CI[0]=0;
    in.r_open (source_name);
    
    for(int i=0;i<inp_act_space_line;i++)in.r_gets(line,BUF_LINE_LENGTH);
    in.r_gets(line,BUF_LINE_LENGTH);
    
    while((key_word_comp(line,act_space_group_start)==0)&&(!in.r_eof())) in.r_gets(line,BUF_LINE_LENGTH);
    if(key_word_comp(line,act_space_group_start)==0){ //active space not found
        fprintf(out_stream,"WARNING: no information about active space in %s\n", source_name);
        fprintf(out_stream,"         [0;0] active space is chosen\n");
        n_act_el_alp[0] = 0;
        n_act_el_bet[0] = 0;
        n_act_orb   [0] = 0;
        mult=1;
        read_CI[0]=0;
        CI[0].get_dim(n_act_orb[0],n_act_el_alp[0], n_act_el_bet[0], 2, mult, print_number, ci_alloc);//two sets are needed for CASSCF
        CI[0].n_states[0]=0;
        CI[0].n_states[1]=0;
    
    }
    else{
        //read size
        n_act_el_alp[0] = kw_to_i(line, n_alp_kw, 0);
        n_act_el_bet[0] = kw_to_i(line, n_bet_kw, 0);
        n_act_orb   [0] = kw_to_i(line, n_act_kw, 0);
        mult = kw_to_i(line, mult_kw, 1);
        print_number = kw_to_i(line, print_number_kw, CAS_PRINT_NUMBER_DEFAULT);
        CI[0].get_dim(n_act_orb[0],n_act_el_alp[0], n_act_el_bet[0], 2, mult, print_number, ci_alloc);//two sets are needed for CASSCF
        CI[0].n_states[0]=0;
        CI[0].n_states[1]=0;
    
        
        
        //reordering
        if(key_word_comp(line, reorder_kw)){
            reorder = kw_to_i(line, reorder_kw, 0);
        }
        if(reorder)if(key_word_comp(line, orbitals_kw)){
            int n=kw_count(line, orbitals_kw,';');
            act_orbs.resize(n);
            kw_to_i_v(&act_orbs, line, orbitals_kw, n);
            fprintf(out_stream,"WARNING:              Orbitals are reordered\n");
            fprintf(out_stream,"                      Active space orbitals:");
            for(int i=0;i<n;i++)
                fprintf(out_stream," %d",act_orbs[i]);
            fprintf(out_stream,"\n");
            if(n!=n_act_orb[0]){
                fprintf(out_stream,"WARNING:              number of reordered orbitals (%d)\n",n);
                fprintf(out_stream,"                      is not equal to the number of active orbitals (%d)\n",n_act_orb[0]);
                fprintf(out_stream,"                      reordering skipped\n");
                reorder=0;
            }
            
        }
                
        if(key_word_comp(line,inp_kw)){
            
            kw_to_s(&CI_file_name[0], line, inp_kw);
            if(read_states==0){
                fprintf(out_stream,"WARNING: CASSCF and CASCI do not need initial CI coefficients\n");
                fprintf(out_stream,"         reading from %s skiped\n",CI_file_name[0]);
            }
            else{
                n_states[0]=kw_count(line, n_st_kw, ';');
                std::vector<int> n_st;
                n_st.resize(n_states[0]);
                kw_to_i_v(&n_st, line, n_st_kw, n_states[0]);
//                 for(int i=0;i<n_st.size();i++)
//                     fprintf(out_stream,"%d ",n_st[i]);
//                 fprintf(out_stream,"\n");
                if(n_states[0]!=0){
                    read_CI[0]=1;
                    CI[0].read_civec(0,CI_file_name[0],n_st);
                    CI[0].n_states[1]=CI[0].n_states[0];//two sets are needed for CASSCF
                }
            }
        }
    }

    in.r_close();
    
    n_cor_orb = (n_el_calc-n_act_el_alp[0]-n_act_el_bet[0])/2;
    if(n_el_calc-n_act_el_alp[0]-n_act_el_bet[0]-2*n_cor_orb){
        fprintf(out_stream,"ERROR: molecule contains odd number of core electrons (n_c = %d)\n", n_el_calc-n_act_el_alp[0]-n_act_el_bet[0]);
        fprintf(out_stream,"       n_el = %d, n_alp = %d, n_bet = %d\n", n_el_calc,n_act_el_alp[0],n_act_el_bet[0]);
        fprintf(out_stream,"       check molecule charge, number of ECP frozen electrons etc.\n");
        exit(1);
    }
    
    
    n_cor_orb_f[0] = n_cor_orb;
    
    if(reorder){
        
        for(int i=0  ;i<n_act_orb[0];i++)
        for(int j=i+1;j<n_act_orb[0];j++)
            if(act_orbs[i]==act_orbs[j]){
                fprintf(out_stream,"ERROR: duplicate orbital in active space\n");
                fprintf(out_stream,"       check numbers of active orbitals\n");
                exit(1);
            }
        
        int n_ch=n_act_orb[0];
        std::vector<int> act_orbs_ch;
        act_orbs_ch.resize(n_act_orb[0]);
        for(int i=0;i<n_act_orb[0];i++)
            act_orbs_ch[i]=1;
        for(int i=0;i<n_act_orb[0];i++)
            if((act_orbs[i]>n_cor_orb)&&(act_orbs[i]<n_cor_orb+n_act_orb[0]+1)){
                act_orbs_ch[act_orbs[i]-n_cor_orb-1]=0;
                n_ch--;
            }
            else
                cv_num.push_back(act_orbs[i]-1);
                
        for(int i=0;i<n_act_orb[0];i++)
            if(act_orbs_ch[i])
                a_num.push_back(i+n_cor_orb);
        
        if(warning){
            fprintf(out_stream,"WARNING:              using orbital reordering with RHF=yes\n");
            fprintf(out_stream,"                      reordering will be done after RHF calculations\n");
        }
        else
            reorder_orbitals();
    }
    
    
//     fprintf(out_stream,"finised\n");
    return 0;
}

int molecule::reorder_orbitals(){
    
    int n_ch = a_num.size();
    int tmp;
//     for(int i=0;i<n_ch;i++)
//         fprintf(out_stream,"%d  \n",a_num[i]+1);
//     getchar();
    
//     for(int i=0;i<n_ch;i++)
//         fprintf(out_stream,"%d  \n",a_num[i]+1);
//     getchar();
    int n_act =n_act_orb[0];
    int * number = new int[n_act];
    for(int i =0; i<n_act;i++) number[i]=i+n_cor_orb;
    
    for(int i=0;i<n_ch;i++){
        change_orbs(MO_VEC, a_num[i],cv_num[i], n_ao);
        tmp               =rep_num[ a_num[i]];
        rep_num[ a_num[i]]=rep_num[cv_num[i]];
        rep_num[cv_num[i]]=tmp           ;
        number[ a_num[i]-n_cor_orb]=cv_num[i];
    }
//     for(int i =0; i<n_act;i++) 
//         fprintf(out_stream,"%d  \n",number[i]);
//     getchar();
    
    for(int i=1;i<n_act;i++){
        if(number[i-1]>number[i]){
            int j=i+n_cor_orb;
//             fprintf(out_stream,"BBBB %d %d  \n",i,number[i]);
//             getchar();
    
            change_orbs(MO_VEC, j, j-1, n_ao);
            tmp        =rep_num[ j];
            rep_num[ j]=rep_num[j-1];
            rep_num[j-1]=tmp        ;
            tmp         =number[i  ];
            number[i  ] =number[i-1] ;
            number[i-1] =tmp         ;
            i=0;
        }
    }
    
//     for(int i =0; i<n_act;i++) 
//         fprintf(out_stream,"A%d  \n",number[i]);
//     getchar();
    
    
    return 0;
}

int molecule::STATES_set_zero(){
    
    n_frag =1;
    
    n_act_el_alp=new int[1];
    n_act_el_bet=new int[1];
    n_act_orb   =new int[1];
    n_states    = new int[1];
    n_cor_orb_f = new int[1];
    
    
    n_states[0]=0;
    n_act_el_alp[0]=0;
    n_act_el_bet[0]=0;
    
    n_cor_orb = (n_el_calc-n_act_el_alp[0]-n_act_el_bet[0])/2;
    if(n_el_calc-n_act_el_alp[0]-n_act_el_bet[0]-2*n_cor_orb){
        fprintf(out_stream,"ERROR: molecule contains odd number of core electrons (n_c = %d)\n", n_el_calc-n_act_el_alp[0]-n_act_el_bet[0]);
        fprintf(out_stream,"       n_el = %d, n_alp = %d, n_bet = %d\n", n_el_calc,n_act_el_alp[0],n_act_el_bet[0]);
        fprintf(out_stream,"       check molecule charge, number of ECP frozen electrons etc.\n");
        exit(1);
    }
    
    n_cor_orb_f[0] = n_cor_orb;
    
    return 0;
}


int molecule::active_space_print(int f){
    
    fprintf(out_stream,"fragment %d\n",f);
    fprintf(out_stream,"n_cor_orb: %d\n",n_cor_orb);
    fprintf(out_stream,"n_act_orb: %d\n",n_act_orb[f]);
    fprintf(out_stream,"n_act_el_alp: %d\n",n_act_el_alp[f]);
    fprintf(out_stream,"n_act_el_bet: %d\n",n_act_el_bet[f]);
    return 0;
}

int molecule::huckel_guess(){
    
    
    n_ro=0;
    
    std::vector<std::vector<double>> lib_coef_r;
    std::vector<int> shell_center_r;
    std::vector<int> core;
    std::vector<double> bf_E;
    
    read_s = basis_lib_read_gbs(this,"huckel/Huckel",1,0,nullptr,nullptr, true, &bf_E, &core);
    for(auto &s: read_s)n_ro+=s.size();
   
    MO_VEC_R = new double[n_ro*n_ro];
//     getchar();
        
    double * SM05   = new double[n_ro*n_ro];
    double * H_huc  = new double[n_ro*n_ro];
    double * E      = new double[n_ro];
    double * S_22   = new double[n_ro*n_ro];
    double * BUF    = new double[std::max(n_ro,n_ao)*std::max(n_ro,n_ao)];
        
    int * core_ao= new int[n_ro]; 
    
    set_zero_matr(H_huc, n_ro*n_ro);
    set_zero_matr(S_22 , n_ro*n_ro);
    AO_1el_from_2shells(S_22,read_s,read_s,n_ro,n_ro,'s',0);
    
    int n_bf=0;
    int sh_size;
    for(int i_s=0; i_s<read_s.size();i_s++){
        sh_size=read_s[i_s].size();
        for(int j=0;j<sh_size;j++){
            H_huc[(n_bf+j)*(n_ro+1)]=-bf_E[i_s];
            core_ao[n_bf+j]=core[i_s];
        }
        n_bf+=sh_size;
        
    }
    
    for(int i=0  ;i<n_ro;i++)
    for(int j=i+1;j<n_ro;j++){
        H_huc[i*n_ro+j]=0.875*S_22[i*n_ro+j]*(H_huc[i*n_ro+i]+H_huc[j*n_ro+j]);
        if(core_ao[i]+core_ao[j])H_huc[i*n_ro+j]=H_huc[i*n_ro+j]*0.05;
        H_huc[j*n_ro+i]=H_huc[i*n_ro+j];
    }
    
    S05_calc(S_22,SM05,BUF,n_ro);
    tr_and_diag(H_huc, SM05, BUF, MO_VEC_R, E, n_ro);
    
    
    delete[] H_huc  ;
    delete[] SM05   ;
    delete[] S_22   ;
    delete[] BUF    ;
    delete[] E      ;
    delete[] core_ao;
    
    
    return 0;
}


int molecule::project_to_full_basis(){    
    
    fprintf(out_stream,"                      orbitals   are projected from f_basis\n");
    fprintf(out_stream,"                      f_basis  = %s\n", read_basis_name);
    double * S_11   = new double[n_ao*n_ao];
    double * S_12   = new double[n_ao*n_ro];
    double * S_22   = new double[n_ro*n_ro];
    double * S_inv  = new double[n_ao*n_ao];
    
    double * BUF    = new double[n_ro*n_ao];
    
    
    set_zero_matr(S_11,n_ao*n_ao);
    set_zero_matr(S_12,n_ao*n_ro);
    set_zero_matr(S_22,n_ro*n_ro);
    
    
    AO_1el_from_2shells(S_22,read_s,read_s,n_ro,n_ro,'s',0);
    AO_1el_from_2shells(S_12,     s,read_s,n_ao,n_ro,'s',0);
    AO_1el_from_2shells(S_11,     s,     s,n_ao,n_ao,'s',0);
    
    set_zero_matr(MO_VEC,n_ao*n_ao);
    memcpy(S_inv,S_11, n_ao*n_ao*sizeof(double));
    inv_matr_constr(S_inv, n_ao);
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasTrans,
                n_ro,n_ao,n_ro,1.0,
                MO_VEC_R,n_ro,
                S_12,n_ro,0.0,
                BUF,n_ao); 
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,  CblasNoTrans,
                std::min(n_ro,n_ao),n_ao,n_ao,1.0,
                BUF,n_ao,
                S_inv,n_ao,0.0,
                MO_VEC,n_ao); 
    
    delete[] S_11   ;
    delete[] S_12   ;
    delete[] S_22   ;
    delete[] S_inv  ;
    delete[] BUF    ;

//     exit(1);
    return 0;
}


int molecule::MO_gen(int guess){
    
    MO_VEC       =new double[n_ao*n_ao];
    NO_VEC       =new double[n_ao*n_ao];
    NO_VEC_B     =new double[n_ao*n_ao];
    orb_energy   =new double[n_ao];
    nat_orb_occ  =new double[n_ao];
    nat_orb_occ_B=new double[n_ao];
    rep_num      =new int   [n_ao];
    
    
    
    recursive_file in;
    char * line = new char[BUF_LINE_LENGTH];
    
    int line_num=0;
    
    int read=1;
    int different_rbasis=0;
    
    in.r_open (source_name);
    for(int i=0;i<inp_mol_line;i++)in.r_gets(line,BUF_LINE_LENGTH);
    in.r_gets(line,BUF_LINE_LENGTH);
    while((key_word_comp(line,MO_group_start)==0)&&(!in.r_eof())){in.r_gets(line,BUF_LINE_LENGTH);}
    if(key_word_comp(line,MO_group_start)==0){
        fprintf(out_stream,"\n");
        MO_VEC_B=MO_VEC;
        read=0;
        different_rbasis=1;
    }    

    

    for(int i=0; i<n_ao;i++)rep_num[i]=-1;
    set_zero_matr(orb_energy,n_ao);
    set_zero_matr(MO_VEC,n_ao*n_ao);
    
    
    if(read){
        in.go_to_begining();
        line_num=0;
        for(int i=0;i<inp_mol_line;i++){in.r_gets(line,BUF_LINE_LENGTH);line_num++;}
        in.r_gets(line,BUF_LINE_LENGTH);line_num++;
        while((key_word_comp(line,read_basis_kw)==0)&&(!in.r_eof())){in.r_gets(line,BUF_LINE_LENGTH);line_num++;}
    
        
        if(key_word_comp(line,read_basis_kw)){
            different_rbasis=1;
            if(read_basis_name  != NULL) delete [] read_basis_name;// kw_to_s will allocate basis_name
            
            if(kw_to_s_w_def(&read_basis_name, line, read_basis_kw, source_name)==0){
                read_basis_in_lib=0;
                read_basis_start_line=line_num;
            }
            else{
                read_basis_in_lib=1;
                read_basis_start_line=0;
            }
            
            read_basis_format=0;//gbs format
            if(strstr(read_basis_name,".exp")){
                read_basis_format=1;//exp format
                if(key_word_comp(line,lib_kw)) read_basis_in_lib=1;
                else read_basis_in_lib=0;
            }
            else
                if (strstr(line,".exp"))read_basis_format=1;
                
            if(strstr(read_basis_name,".gbs")){
                read_basis_format=0;//gbs format
                if(key_word_comp(line,lib_kw)) read_basis_in_lib=1;
                else read_basis_in_lib=0;
            }
            
            if(read_basis_format==0)
                read_s=basis_lib_read_gbs(this,read_basis_name,
                                           read_basis_in_lib,read_basis_start_line,
                                           nullptr,nullptr, false,nullptr,nullptr);
            if(read_basis_format==1)
                read_s=basis_lib_read_exp(this,read_basis_name,
                                           read_basis_in_lib,read_basis_start_line,
                                           nullptr,nullptr, false);
                
            
        }
        else{
            different_rbasis=0;
            read_s = s;
        }
        
        n_ro=0;
        for(auto &s: read_s)n_ro+=s.size();
        MO_VEC_R=new double[n_ro*n_ro];
        UHF_MO_read();
        MO_libint_format();
    }
    else{
        if(read_basis_name  != NULL) delete [] read_basis_name;
        read_basis_name=new char[BUF_LINE_LENGTH];

        int have_guess=0;
        if      (guess==GUESS_HUCKEL) have_guess=0;
        else if (guess==GUESS_SAD   ) have_guess=sad_guess(this);
        else{
            fprintf(out_stream,"ERROR: unknown starting guess (%d)\n",guess);
            exit(1);
        }

        if(have_guess==0){
            fprintf(out_stream,"                      using Huckel orbitals\n");
            sprintf(read_basis_name,"%s","huckel/Huckel");
            huckel_guess();
        }
        fprintf(out_stream,"\n\n");
//         printf_timer("Huckel guess");

    }
    if(different_rbasis)
        project_to_full_basis();
    else
        memcpy(MO_VEC, MO_VEC_R, n_ao*n_ao*sizeof(double));
    
    if(n_ro!=n_ao){
        delete[] MO_VEC_R;
        n_ro=n_ao;
        MO_VEC_R=new double[n_ao*n_ao];
    }
    
    return 0;
}

int molecule::SCF_type_def(){
    
    FILE *in;
    char tmpstr[BUF_LINE_LENGTH];
    
    in = fopen (source_name,"r");
    
    fgets(tmpstr,BUF_LINE_LENGTH,in);
    while((key_word_comp(tmpstr,wf_type)==0)&&(key_word_comp(tmpstr,act_space_group_start)==0)&&(!feof(in))) fgets(tmpstr,BUF_LINE_LENGTH,in);
    
    fclose(in);
    
    if(key_word_comp(tmpstr,uhf)) return 1;
    
    return 0;
    
}

int molecule::UHF_MO_read(){
    
    recursive_file in;
    char * line = new char[BUF_LINE_LENGTH];
    
    int start;
    
    
    in.r_open (source_name);
    for(int i=0;i<inp_mol_line;i++)in.r_gets(line,BUF_LINE_LENGTH);
    in.r_gets(line,BUF_LINE_LENGTH);
    while((key_word_comp(line,MO_group_start)==0)&&(!in.r_eof())){in.r_gets(line,BUF_LINE_LENGTH);}
//     char * vec_file = NULL;
    if(key_word_comp(line,inp_kw)){
        kw_to_s(&MO_name, line, inp_kw);
        fprintf(out_stream,"                      included   %s (orbitals)\n",MO_name);
        start=0;
    }
    else{
        MO_name=new char[BUF_LINE_LENGTH];
        sprintf(MO_name,"%s\0",source_name);
        start=inp_mol_line;
    }
    in.r_close();
    delete[] line   ;
    
    int binary_MO=0;
    if(strstr(MO_name,".orb"))binary_MO=1;
    if(strstr(MO_name,".orb_GAMESS"))binary_MO=0;
    
    if(binary_MO){
        n_mo=binary_VEC_read(MO_name,MO_VEC_R,orb_energy,n_ro,n_ro,MO_VEC_B,orb_energy_B);
        
    }
    else{
        int tmp_n_mo=VEC_read(MO_name, start, MO_VEC_R, n_ro, n_ro, 'a'); //number of orbitals in input may be less than n_ro
        if(tmp_n_mo==-18){
            tmp_n_mo=molden_read();
//             n_mo=tmp_n_mo;
        }
        if(n_mo>tmp_n_mo)n_mo=tmp_n_mo;
        energy_read(MO_name, start, orb_energy, n_mo,'a');
        
        
        if(SCF_type_def()){
            printf("UHF wavefunctions are not supported\n");
            exit(1);
//             MO_VEC_B=new double[n_ao*n_ao];
//             orb_energy_B=new double[n_ao];
//             set_zero_matr(orb_energy_B,n_ao);
//             VEC_read(MO_name, MO_VEC_B, n_ao, n_ao,'b');
//             energy_read(MO_name, orb_energy_B, n_mo,'a');
        }
        else{
            MO_VEC_B=MO_VEC;
            orb_energy_B=orb_energy;
        }
    }
        
    return 0;
}

int molecule::molden_read(){
    
    
    
    set_zero_matr(MO_VEC_R,n_ro*n_ro);
    recursive_file  in;
    in.r_open (MO_name);
    char * line = new char[BUF_LINE_LENGTH];
    int line_num=0;
    for(int i=0;i<inp_mol_line;i++)in.r_gets(line,BUF_LINE_LENGTH);
    in.r_smart_fgets(line,&line_num,MO_name, "serching [MO] in molden file");
    while((strstr(line,"[MO]")==NULL)&&(!in.r_eof())) in.r_smart_fgets(line,&line_num,MO_name, "serching [MO] in molden file");
//     printf("AAAAAA - %s\n", line);
//     getchar();
    int molden_D5=0;
    int molden_F7=0;
    int molden_G9=0;
//     if(strstr(line,"D6"))molden_D6=1;
    
    double c;
    int i;
    int tmp_int;
    in.r_gets(line,BUF_LINE_LENGTH);//fprintf(out_stream,"%s",line);
    for(i=0;i<n_ro;i++){
        in.r_gets(line,BUF_LINE_LENGTH);//fprintf(out_stream,"%s",line);
        in.r_gets(line,BUF_LINE_LENGTH);//fprintf(out_stream,"%s",line);
        in.r_gets(line,BUF_LINE_LENGTH);//fprintf(out_stream,"%s",line);
        if(strstr(line,"Occup=")==nullptr)break;
//         getchar();
        int k=0;
        for(auto &AO: read_s){
            if(AO.contr[0].l==0){
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k);
//                 fprintf(out_stream,"S(%d) - %d  %f %s",AO.contr[0].l,k,MO_VEC_R[i*n_ro+k],line);getchar();
                k++;
                continue;
            }
            if(AO.contr[0].l==1){
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k);
                //fprintf(out_stream,"P(%d) - %d  %f %s",AO.contr[0].l,k,MO_VEC_R[i*n_ro+k],line);getchar();
                k++;
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k);
                //fprintf(out_stream,"P(%d) - %d  %f %s",AO.contr[0].l,k,MO_VEC_R[i*n_ro+k],line);getchar();
                k++;
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k);
                //fprintf(out_stream,"P(%d) - %d  %f %s",AO.contr[0].l,k,MO_VEC_R[i*n_ro+k],line);getchar();
                k++;
                continue;
            }
            if(AO.contr[0].l==2)if(molden_D5==1){
//                 fprintf(out_stream,"d\n");
                MO_VEC_R[i*n_ro+k  ]=0;
                MO_VEC_R[i*n_ro+k+1]=0;
                MO_VEC_R[i*n_ro+k+2]=0;
                MO_VEC_R[i*n_ro+k+3]=0;
                MO_VEC_R[i*n_ro+k+4]=0;
                MO_VEC_R[i*n_ro+k+5]=0;
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,&c);
                //fprintf(out_stream,"D(%d) - %d  %f %s",AO.contr[0].l,k,c,line);getchar();
                MO_VEC_R[i*n_ro+k  ]+=   -c/sqrt(4.0);//d0 to xx
                MO_VEC_R[i*n_ro+k+1]+=   -c/sqrt(4.0);//d0 to yy
                MO_VEC_R[i*n_ro+k+2]+=2.0*c/sqrt(4.0);//d0 to zz
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,&c);
                //fprintf(out_stream,"D(%d) - %d  %f %s",AO.contr[0].l,k+1,c,line);getchar();
                MO_VEC_R[i*n_ro+k+4]+=c;//xz
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,&c);
                //fprintf(out_stream,"D(%d) - %d  %f %s",AO.contr[0].l,k+2,c,line);getchar();
                MO_VEC_R[i*n_ro+k+5]+=c;//yz
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,&c);
                //fprintf(out_stream,"D(%d) - %d  %f %s",AO.contr[0].l,k+3,c,line);
                MO_VEC_R[i*n_ro+k  ]+= c/sqrt(4.0/3.0);//d2 to xx
                MO_VEC_R[i*n_ro+k+1]+=-c/sqrt(4.0/3.0);//d2 to xx
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,&c);
                MO_VEC_R[i*n_ro+k+3]+=c;//xy
                //fprintf(out_stream,"D(%d) - %d  %f %s",AO.contr[0].l,k+4,c,line);
                
                k+=6;
                continue;
            }
            if(AO.contr[0].l==2)if(molden_D5==0){
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k  );//xx
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+1);//yy
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+2);//zz
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+3);//xy
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+4);//yz
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+5);//xz
//                 MO_VEC_R[i*n_ro+k+3]=MO_VEC_R[i*n_ro+k+3]/sqrt(3.0);
//                 MO_VEC_R[i*n_ro+k+4]=MO_VEC_R[i*n_ro+k+4]/sqrt(3.0);
//                 MO_VEC_R[i*n_ro+k+5]=MO_VEC_R[i*n_ro+k+5]/sqrt(3.0);
                k+=6;
                continue;
            }
            if(AO.contr[0].l==3)if(molden_F7==1){
//                 fprintf(out_stream,"f\n");
                MO_VEC_R[i*n_ro+k  ]=0;
                MO_VEC_R[i*n_ro+k+1]=0;
                MO_VEC_R[i*n_ro+k+2]=0;
                MO_VEC_R[i*n_ro+k+3]=0;
                MO_VEC_R[i*n_ro+k+4]=0;
                MO_VEC_R[i*n_ro+k+5]=0;
                MO_VEC_R[i*n_ro+k+6]=0;
                MO_VEC_R[i*n_ro+k+7]=0;
                MO_VEC_R[i*n_ro+k+8]=0;
                MO_VEC_R[i*n_ro+k+9]=0;
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,&c);
                //fprintf(out_stream,"F - %d  %f %s",k,c,line);
                MO_VEC_R[i*n_ro+k  ]+=0 *c/sqrt(4.0)           ;
                MO_VEC_R[i*n_ro+k+1]+=0 *c/sqrt(4.0)           ;
                MO_VEC_R[i*n_ro+k+2]+=2 *c/sqrt(4.0)           ;
                MO_VEC_R[i*n_ro+k+3]+=0 *c/sqrt(4.0)/sqrt( 5.0);
                MO_VEC_R[i*n_ro+k+4]+=-3*c/sqrt(4.0)/sqrt( 5.0);
                MO_VEC_R[i*n_ro+k+5]+=0 *c/sqrt(4.0)/sqrt( 5.0);
                MO_VEC_R[i*n_ro+k+6]+=0 *c/sqrt(4.0)/sqrt( 5.0);
                MO_VEC_R[i*n_ro+k+7]+=-3*c/sqrt(4.0)/sqrt( 5.0);
                MO_VEC_R[i*n_ro+k+8]+=0 *c/sqrt(4.0)/sqrt( 5.0);
                MO_VEC_R[i*n_ro+k+9]+=0 *c/sqrt(4.0)/sqrt(15.0);
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,&c);
                ////fprintf(out_stream,"%d  %f %s",k+1,c,line);
                MO_VEC_R[i*n_ro+k  ]+=-1*c/sqrt(8.0/3.0)           ;//XXX
                MO_VEC_R[i*n_ro+k+1]+=0 *c/sqrt(8.0/3.0)           ;//YYY
                MO_VEC_R[i*n_ro+k+2]+=0 *c/sqrt(8.0/3.0)           ;//ZZZ
                MO_VEC_R[i*n_ro+k+3]+=0 *c/sqrt(8.0/3.0)/sqrt( 5.0);//XXY
                MO_VEC_R[i*n_ro+k+4]+=0 *c/sqrt(8.0/3.0)/sqrt( 5.0);//XXZ
                MO_VEC_R[i*n_ro+k+5]+=-1*c/sqrt(8.0/3.0)/sqrt( 5.0);//XYY
                MO_VEC_R[i*n_ro+k+6]+=4 *c/sqrt(8.0/3.0)/sqrt( 5.0);//XZZ
                MO_VEC_R[i*n_ro+k+7]+=0 *c/sqrt(8.0/3.0)/sqrt( 5.0);//YYZ
                MO_VEC_R[i*n_ro+k+8]+=0 *c/sqrt(8.0/3.0)/sqrt( 5.0);//YZZ
                MO_VEC_R[i*n_ro+k+9]+=0 *c/sqrt(8.0/3.0)/sqrt(15.0);//XYZ
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,&c);
                ////fprintf(out_stream,"%d  %f %s",k+2,c,line);
                MO_VEC_R[i*n_ro+k  ]+=0 *c/sqrt(8.0/3.0)           ;//XXX
                MO_VEC_R[i*n_ro+k+1]+=-1*c/sqrt(8.0/3.0)           ;//YYY
                MO_VEC_R[i*n_ro+k+2]+=0 *c/sqrt(8.0/3.0)           ;//ZZZ
                MO_VEC_R[i*n_ro+k+3]+=-1*c/sqrt(8.0/3.0)/sqrt( 5.0);//XXY
                MO_VEC_R[i*n_ro+k+4]+=0 *c/sqrt(8.0/3.0)/sqrt( 5.0);//XXZ
                MO_VEC_R[i*n_ro+k+5]+=0 *c/sqrt(8.0/3.0)/sqrt( 5.0);//XYY
                MO_VEC_R[i*n_ro+k+6]+=0 *c/sqrt(8.0/3.0)/sqrt( 5.0);//XZZ
                MO_VEC_R[i*n_ro+k+7]+=0 *c/sqrt(8.0/3.0)/sqrt( 5.0);//YYZ
                MO_VEC_R[i*n_ro+k+8]+=4 *c/sqrt(8.0/3.0)/sqrt( 5.0);//YZZ
                MO_VEC_R[i*n_ro+k+9]+=0 *c/sqrt(8.0/3.0)/sqrt(15.0);//XYZ
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,&c);
                ////fprintf(out_stream,"%d  %f %s",k+3,c,line);
                MO_VEC_R[i*n_ro+k  ]+= 0*c/sqrt(8.0/30.0)           ;//XXX
                MO_VEC_R[i*n_ro+k+1]+= 0*c/sqrt(8.0/30.0)           ;//YYY
                MO_VEC_R[i*n_ro+k+2]+= 0*c/sqrt(8.0/30.0)           ;//ZZZ
                MO_VEC_R[i*n_ro+k+3]+= 0*c/sqrt(8.0/30.0)/sqrt( 5.0);//XXY
                MO_VEC_R[i*n_ro+k+4]+= 1*c/sqrt(8.0/30.0)/sqrt( 5.0);//XXZ
                MO_VEC_R[i*n_ro+k+5]+= 0*c/sqrt(8.0/30.0)/sqrt( 5.0);//XYY
                MO_VEC_R[i*n_ro+k+6]+= 0*c/sqrt(8.0/30.0)/sqrt( 5.0);//XZZ
                MO_VEC_R[i*n_ro+k+7]+=-1*c/sqrt(8.0/30.0)/sqrt( 5.0);//YYZ
                MO_VEC_R[i*n_ro+k+8]+= 0*c/sqrt(8.0/30.0)/sqrt( 5.0);//YZZ
                MO_VEC_R[i*n_ro+k+9]+= 0*c/sqrt(8.0/30.0)/sqrt(15.0);//XYZ
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,&c);
                ////fprintf(out_stream,"%d  %f %s",k+4,c,line);
                MO_VEC_R[i*n_ro+k  ]+= 0 *c/sqrt(1.0/15.0)           ;//XXX
                MO_VEC_R[i*n_ro+k+1]+= 0 *c/sqrt(1.0/15.0)           ;//YYY
                MO_VEC_R[i*n_ro+k+2]+= 0 *c/sqrt(1.0/15.0)           ;//ZZZ
                MO_VEC_R[i*n_ro+k+3]+= 0 *c/sqrt(1.0/15.0)/sqrt( 5.0);//XXY
                MO_VEC_R[i*n_ro+k+4]+= 0 *c/sqrt(1.0/15.0)/sqrt( 5.0);//XXZ
                MO_VEC_R[i*n_ro+k+5]+= 0 *c/sqrt(1.0/15.0)/sqrt( 5.0);//XYY
                MO_VEC_R[i*n_ro+k+6]+= 0 *c/sqrt(1.0/15.0)/sqrt( 5.0);//XZZ
                MO_VEC_R[i*n_ro+k+7]+= 0 *c/sqrt(1.0/15.0)/sqrt( 5.0);//YYZ
                MO_VEC_R[i*n_ro+k+8]+= 0 *c/sqrt(1.0/15.0)/sqrt( 5.0);//YZZ
                MO_VEC_R[i*n_ro+k+9]+= 1 *c/sqrt(1.0/15.0)/sqrt(15.0);//XYZ
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,&c);
                ////fprintf(out_stream,"%d  %f %s",k+5,c,line);
                MO_VEC_R[i*n_ro+k  ]+= 1*c/sqrt(1.6)           ;//XXX
                MO_VEC_R[i*n_ro+k+1]+= 0*c/sqrt(1.6)           ;//YYY
                MO_VEC_R[i*n_ro+k+2]+= 0*c/sqrt(1.6)           ;//ZZZ
                MO_VEC_R[i*n_ro+k+3]+= 0*c/sqrt(1.6)/sqrt( 5.0);//XXY
                MO_VEC_R[i*n_ro+k+4]+= 0*c/sqrt(1.6)/sqrt( 5.0);//XXZ
                MO_VEC_R[i*n_ro+k+5]+=-3*c/sqrt(1.6)/sqrt( 5.0);//XYY
                MO_VEC_R[i*n_ro+k+6]+= 0*c/sqrt(1.6)/sqrt( 5.0);//XZZ
                MO_VEC_R[i*n_ro+k+7]+= 0*c/sqrt(1.6)/sqrt( 5.0);//YYZ
                MO_VEC_R[i*n_ro+k+8]+= 0*c/sqrt(1.6)/sqrt( 5.0);//YZZ
                MO_VEC_R[i*n_ro+k+9]+= 0*c/sqrt(1.6)/sqrt(15.0);//XYZ
                in.r_gets(line,BUF_LINE_LENGTH);
                sscanf(line,"%d %lf", &tmp_int,&c);
                ////fprintf(out_stream,"%d  %f %s",k+6,c,line);
                MO_VEC_R[i*n_ro+k  ]+= 0*c/sqrt(1.6)           ;//XXX
                MO_VEC_R[i*n_ro+k+1]+=-1*c/sqrt(1.6)           ;//YYY
                MO_VEC_R[i*n_ro+k+2]+= 0*c/sqrt(1.6)           ;//ZZZ
                MO_VEC_R[i*n_ro+k+3]+= 3*c/sqrt(1.6)/sqrt( 5.0);//XXY
                MO_VEC_R[i*n_ro+k+4]+= 0*c/sqrt(1.6)/sqrt( 5.0);//XXZ
                MO_VEC_R[i*n_ro+k+5]+= 0*c/sqrt(1.6)/sqrt( 5.0);//XYY
                MO_VEC_R[i*n_ro+k+6]+= 0*c/sqrt(1.6)/sqrt( 5.0);//XZZ
                MO_VEC_R[i*n_ro+k+7]+= 0*c/sqrt(1.6)/sqrt( 5.0);//YYZ
                MO_VEC_R[i*n_ro+k+8]+= 0*c/sqrt(1.6)/sqrt( 5.0);//YZZ
                MO_VEC_R[i*n_ro+k+9]+= 0*c/sqrt(1.6)/sqrt(15.0);//XYZ
                k+=10;
                continue;
            }
            if(AO.contr[0].l==3)if(molden_F7==0){// see https://www.theochem.ru.nl/molden/molden_format.html
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k  );//XXX ^
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+1);//YYY |
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+2);//ZZZ |
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+5);//YYX???
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+3);//XXY
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+4);//XXZ
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+7);//ZZX
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+8);//ZZY
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+6);//YYZ
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+9);//XYZ
//                 MO_VEC_R[i*n_ro+k+3]=MO_VEC_R[i*n_ro+k+3]/sqrt( 5.0);
//                 MO_VEC_R[i*n_ro+k+4]=MO_VEC_R[i*n_ro+k+4]/sqrt( 5.0);
//                 MO_VEC_R[i*n_ro+k+5]=MO_VEC_R[i*n_ro+k+5]/sqrt( 5.0);
//                 MO_VEC_R[i*n_ro+k+3]=MO_VEC_R[i*n_ro+k+6]/sqrt( 5.0);
//                 MO_VEC_R[i*n_ro+k+4]=MO_VEC_R[i*n_ro+k+7]/sqrt( 5.0);
//                 MO_VEC_R[i*n_ro+k+5]=MO_VEC_R[i*n_ro+k+8]/sqrt( 5.0);
//                 MO_VEC_R[i*n_ro+k+3]=MO_VEC_R[i*n_ro+k+9]/sqrt(15.0);
                
                k+=10;
                continue;
            }
            if(AO.contr[0].l==4)if(molden_G9==0){
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k   );//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+ 1);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+ 2);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+ 3);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+ 4);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+ 5);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+ 6);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+ 7);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+ 8);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+ 9);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+10);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+11);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+12);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+13);//printf("%s\n",line);
                in.r_gets(line,BUF_LINE_LENGTH);sscanf(line,"%d %lf", &tmp_int,MO_VEC_R+i*n_ro+k+14);//printf("%s\n",line);
                
                k+=15;
                continue;
            }
            if(AO.contr[0].l==4)if(molden_G9==1){
                fprintf(out_stream,"ERROR: molden_read is not realized for spherical G9\n");
                exit(1);
            }
            if(AO.contr[0].l>4){
                fprintf(out_stream,"ERROR: molden_read is not realized for orbital higher than G\n");
                exit(1);
            }
            
        }
        // printf("%d\n",k);
        // getchar();
        in.r_gets(line,BUF_LINE_LENGTH);//fprintf(out_stream,"%s",line);
        if(key_word_comp(line,MO_group_end)){
            i++;
            break;
        }
    }
    
//     fprintf(out_stream,"%d\n",i);
//     exit(0);
    
    delete[] line;
    
    return i;
}

int GAMESS_VEC_data_print(FILE * out, double * V, int n_mo, int n_ao){
    
    int i,j,k,l;
    for(i=0;i<n_mo;i++){
        for(j=0;j<n_ao/5;j++)
            fprintf(out,"%2d%3d% .8E% .8E% .8E% .8E% .8E\n",(i+1)%100
                                                           ,(j+1)%1000
                                                           ,V[i*n_ao+5*j+0]
                                                           ,V[i*n_ao+5*j+1]
                                                           ,V[i*n_ao+5*j+2]
                                                           ,V[i*n_ao+5*j+3]
                                                           ,V[i*n_ao+5*j+4]);
        l = n_ao - 5*j;
        if(l){
//             j++;
            fprintf(out,"%2d%3d",(i+1)%100,(j+1)%1000);
            for(k=0;k<l;k++)fprintf(out,"% .8E",V[i*n_ao+5*j+k]);
            fprintf(out,"\n");
        }
    }
    
    return 0;
}

int orb_energy_print(FILE * out, double * e, int n_ao){
    
    int i;
    for(i=0;i<n_ao;i++)
        fprintf(out,"% .8E\n",e[i]);
    return 0;
}

int molecule::MO_print(const char * out_name){
    
//     if(MO_VEC_B==MO_VEC)
        binary_VEC_write(out_name,MO_VEC_R,orb_energy,n_mo,n_ao,NULL,NULL);
//     if(MO_VEC_B!=MO_VEC)
//         binary_VEC_write(out_name,MO_VEC_R,orb_energy,n_mo,n_ao,MO_VEC_B,orb_energy_B);
    
    char * name = new char[BUF_LINE_LENGTH];
    sprintf(name,"%s_GAMESS\0",out_name);
    
    
    
    FILE * out;
    out = fopen(name,"w");
    
    fprintf(out," $VEC\n");
    GAMESS_VEC_data_print(out,MO_VEC_R,n_mo,n_ao);//realy n_mo
    fprintf(out," $VECEND\n");
    fprintf(out," $ENERGY\n");
    orb_energy_print(out,orb_energy,n_mo);//realy n_mo
    fprintf(out," $ENERGYEND\n");
    
    if(MO_VEC_B!=MO_VEC){
        printf("MO_print does not support UHF orbitals\n");
//         fprintf(out," $WF_TYPE UHF WF_TYPE_END$\n");
//         fprintf(out," $VEC_B\n");
//         GAMESS_VEC_data_print(out,MO_VEC_B,n_mo,n_ao);//realy n_mo
//         fprintf(out," $VECEND\n");
//         fprintf(out," $ENERGY_B\n");
//         orb_energy_print(out,orb_energy_B,n_mo);//realy n_mo
//         fprintf(out," $ENERGYEND\n");
    }
    else{
        fprintf(out," $WF_TYPE RHF WF_TYPE_END$\n");
    }
    
    fclose(out);
    delete[] name;
    
    return 0;
}

int VEC_energy_cpy(double * O_V,double * I_V, double * o_e, double * i_e, int n){
    
    for(int k=0;k<n*n;k++) O_V[k]=I_V[k];
    for(int k=0;k<n  ;k++) o_e[k]=i_e[k];
    return 0;
}

int molecule::write_orb_index(){
    
    fprintf(out_stream,"write_orb_index is deprecated\n");
    exit(0);
    
//     int i=0;
//     while(i<n_cor_orb){
//         orb_energy[i]=1.0;
//         i++;
//     }
//     while(i<n_cor_orb+n_act_orb[0]){
//         orb_energy[i]=0.0;
//         i++;
//     }
//     while(i<n_ao){
//         orb_energy[i]=-1.0;
//         i++;
//     }
//     
//     return 0;
    
}

int molecule::NO_print(const char * name, char S){
    if(S!='T'){
        fprintf(out_stream,"NO_print is written only for total spin\n");
        exit(0);
    }
    double * MO_backup = new double[n_ao*(n_cor_orb+n_act_orb[0])]; 
    double *  e_backup = new double[      n_cor_orb+n_act_orb[0] ]; 
    int    * rn_backup = new int   [      n_cor_orb+n_act_orb[0] ]; 
    memcpy(MO_backup, MO_VEC    ,sizeof(double)*n_ao*(n_cor_orb+n_act_orb[0]));
    memcpy( e_backup, orb_energy,sizeof(double)*     (n_cor_orb+n_act_orb[0]));
    memcpy(rn_backup, rep_num   ,sizeof(int   )*     (n_cor_orb+n_act_orb[0]));
    // memcpy(MO_VEC   , NO_VEC,sizeof(double)*n_ao*(n_cor_orb+n_act_orb[0]));
    
    int i=0;
    int i_no=n_act_orb[0]-1;
    while(i<n_cor_orb){
        orb_energy[i]=2.0;
        rep_num   [i]=-1 ;
        i++;
    }
    while(i<n_cor_orb+n_act_orb[0]){
        orb_energy[i]=nat_orb_occ[i_no];
        rep_num   [i]=-1;
        for(int j = 0;j< n_ao;j++)
            MO_VEC[i*n_ao+j]=NO_VEC[(i_no+n_cor_orb)*n_ao+j]*sqrt(2.0/nat_orb_occ[i_no]);
        i++;
        i_no--;
    }
    
    check_orb_symmetry();
    MO_gamess_format();
    
    char * file_name = new char[BUF_LINE_LENGTH];;
    sprintf(file_name,"%s_NatOrb.out\0",name);
    GAMESS_type_out_print(file_name, n_cor_orb+n_act_orb[0]);
    fprintf(out_stream,"visualization file: %s\n",file_name);
    
    sprintf(file_name,"%s_NatOrb.orb\0",name);
    int n_mo_backup = n_mo;
    // n_mo=n_cor_orb+n_act_orb[0];
    MO_print(file_name);
    // n_mo = n_mo_backup;
    fprintf(out_stream,"data file         : %s\n",file_name);
        
    
    memcpy(MO_VEC    , MO_backup, sizeof(double)*n_ao*(n_cor_orb+n_act_orb[0]));
    memcpy(orb_energy,  e_backup, sizeof(double)*     (n_cor_orb+n_act_orb[0]));
    memcpy(rep_num   , rn_backup, sizeof(int   )*     (n_cor_orb+n_act_orb[0]));

    delete[] MO_backup;
    delete[]  e_backup;

    return 0;

}

int molecule::LOC_print(const char * name, const double * U){
    const int n_act = n_act_orb[0];
    const int n_cor = n_cor_orb;
    double * MO_backup = new double[n_ao*n_act];
    double *  e_backup = new double[     n_act];
    int    * rn_backup = new int   [     n_act];
    memcpy(MO_backup, MO_VEC     + n_cor*n_ao, sizeof(double)*n_ao*n_act);
    memcpy( e_backup, orb_energy + n_cor     , sizeof(double)*     n_act);
    memcpy(rn_backup, rep_num    + n_cor     , sizeof(int   )*     n_act);

    // localized active orbitals (rows n_cor..n_cor+n_act): C_loc[p][ao] = sum_a C_act[a][ao] U[a,p],
    // read from the backed-up active block so the in-place overwrite below is safe. They carry no
    // canonical energy/irrep, so flag them (energy 0, rep -1).
    for(int p=0;p<n_act;p++){
        orb_energy[n_cor+p]=0.0;
        rep_num   [n_cor+p]=-1;
    }
    nopt_par_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                   n_act, n_ao, n_act, 1.0,
                   U, n_act,
                   MO_backup, n_ao,0.0,
                   MO_VEC+n_cor*n_ao,n_ao);

    check_orb_symmetry();
    MO_gamess_format();

    char * file_name = new char[BUF_LINE_LENGTH];
    sprintf(file_name,"%s_LocOrb.out",name);
    GAMESS_type_out_print(file_name, n_cor+n_act);
    fprintf(out_stream,"visualization file: %s\n",file_name);
    sprintf(file_name,"%s_LocOrb.orb",name);
    MO_print(file_name);
    fprintf(out_stream,"data file         : %s\n",file_name);
    delete[] file_name;

    memcpy(MO_VEC     + n_cor*n_ao, MO_backup, sizeof(double)*n_ao*n_act);
    memcpy(orb_energy + n_cor     ,  e_backup, sizeof(double)*     n_act);
    memcpy(rep_num    + n_cor     , rn_backup, sizeof(int   )*     n_act);
    delete[] MO_backup;
    delete[]  e_backup;
    delete[] rn_backup;
    return 0;
}


int molecule::GAMESS_type_out_print(const char* out_name, int n_print_mo){
    
    int i,j,k;
    char type_shell='X';
    int gto_num=1;
    int shell_num=1;
    int ao_num=0;
    FILE * out;
    out = fopen(out_name,"w");
//     out = stderr;
//     b= &(basis);
    int * atom_num = new int[n_ao];
    for(int i=0;i<n_ao;i++)atom_num[i]=-1;
    char * ao_type=new char[3*n_ao];
    
    if(n_print_mo==-1)n_print_mo=n_mo;
    
    if(out==NULL){fprintf(out_stream,"couldn't open file %s to print structure\n",out_name);
                  return 1;}
    
    fprintf(out," GAMESS VERSIONS\n");
    fprintf(out," ATOM      ATOMIC                      COORDINATES (BOHR)\n");
    fprintf(out,"           CHARGE         X                   Y                   Z\n");
    for(i=0;i<n_atoms;i++)if(nucl_charges_full[i])
    fprintf(out,"%c%c          %.1f % 16.10f    % 16.10f    % 16.10f\n",
                atom_names[i][0],atom_names[i][1],
                nucl_charges_full[i],
                atom_coord[3*i  ],atom_coord[3*i+1],atom_coord[3*i+2]);
    for(i=0;i<n_atoms;i++)if(nucl_charges_full[i]==0)
    fprintf(out,"%c%c          %.1f % 16.10f    % 16.10f    % 16.10f\n",
                atom_names[i][0],atom_names[i][1],
                nucl_charges_full[i],
                atom_coord[3*i  ],atom_coord[3*i+1],atom_coord[3*i+2]);
    fprintf(out,"\n");
    fprintf(out,"     ATOMIC BASIS SET\n");
    fprintf(out,"     ----------------\n");
    fprintf(out," THE CONTRACTED PRIMITIVE FUNCTIONS HAVE BEEN UNNORMALIZED\n");
    fprintf(out," THE CONTRACTED BASIS FUNCTIONS ARE NOW NORMALIZED TO UNITY\n");

    
    
    
//     for(auto &a: shell_center){
//         fprintf(out_stream,"%d\n",a);
//     }
//     getchar();
//     
    
    fprintf(out,"\n");
    fprintf(out," SHELL TYPE PRIM    EXPONENT          CONTRACTION COEFFICIENTS\n\n");
    for(i=0;i<n_atoms;i++)if(nucl_charges_full[i]>0){//virtual atoms must be added
        fprintf(out," %c%c        \n",atom_names[i][0],atom_names[i][1]);
        fprintf(out,"\n");
        
        for(int sh_i=0;sh_i<s.size();sh_i++)
        if(shell_center[sh_i]==i){
            for(int j=0;j<s[sh_i].contr[0].size();j++){
                atom_num[ao_num+j]=i;
            }
            if(s[sh_i].contr[0].l==0){type_shell='S';ao_type[3*ao_num]=' ';ao_type[3*ao_num+1]=' ';ao_type[3*ao_num+2]='S';}
            if(s[sh_i].contr[0].l==1){
                type_shell='P';
                ao_type[3*ao_num  ]=' ';ao_type[3*ao_num+1]=' ';ao_type[3*ao_num+2]='X';
                ao_type[3*ao_num+3]=' ';ao_type[3*ao_num+4]=' ';ao_type[3*ao_num+5]='Y';
                ao_type[3*ao_num+6]=' ';ao_type[3*ao_num+7]=' ';ao_type[3*ao_num+8]='Z';
            }
            if(s[sh_i].contr[0].l==2){
                type_shell='D';
                ao_type[3*ao_num   ]=' ';ao_type[3*ao_num+1 ]='X';ao_type[3*ao_num+2 ]='X';
                ao_type[3*ao_num+3 ]=' ';ao_type[3*ao_num+4 ]='Y';ao_type[3*ao_num+5 ]='Y';
                ao_type[3*ao_num+6 ]=' ';ao_type[3*ao_num+7 ]='Z';ao_type[3*ao_num+8 ]='Z';
                ao_type[3*ao_num+9 ]=' ';ao_type[3*ao_num+10]='X';ao_type[3*ao_num+11]='Y';
                ao_type[3*ao_num+12]=' ';ao_type[3*ao_num+13]='X';ao_type[3*ao_num+14]='Z';
                ao_type[3*ao_num+15]=' ';ao_type[3*ao_num+16]='Y';ao_type[3*ao_num+17]='Z';
            }
            if(s[sh_i].contr[0].l==3){
                type_shell='F';
                ao_type[3*ao_num   ]='X';ao_type[3*ao_num+1 ]='X';ao_type[3*ao_num+2 ]='X';
                ao_type[3*ao_num+3 ]='Y';ao_type[3*ao_num+4 ]='Y';ao_type[3*ao_num+5 ]='Y';
                ao_type[3*ao_num+6 ]='Z';ao_type[3*ao_num+7 ]='Z';ao_type[3*ao_num+8 ]='Z';
                ao_type[3*ao_num+9 ]='X';ao_type[3*ao_num+10]='X';ao_type[3*ao_num+11]='Y';
                ao_type[3*ao_num+12]='X';ao_type[3*ao_num+13]='X';ao_type[3*ao_num+14]='Z';
                ao_type[3*ao_num+15]='Y';ao_type[3*ao_num+16]='Y';ao_type[3*ao_num+17]='X';
                ao_type[3*ao_num+18]='Y';ao_type[3*ao_num+19]='Y';ao_type[3*ao_num+20]='Z';
                ao_type[3*ao_num+21]='Z';ao_type[3*ao_num+22]='Z';ao_type[3*ao_num+23]='X';
                ao_type[3*ao_num+24]='Z';ao_type[3*ao_num+25]='Z';ao_type[3*ao_num+26]='Y';
                ao_type[3*ao_num+27]='X';ao_type[3*ao_num+28]='Y';ao_type[3*ao_num+29]='Z';
            }
            if(s[sh_i].contr[0].l==4){
                type_shell='G';
                for(int iiii=3*ao_num; iiii<3*ao_num+45;iiii++)ao_type[iiii]='G';
            }
            if(s[sh_i].contr[0].l==5){
                type_shell='H';
                for(int iiii=3*ao_num; iiii<3*ao_num+63;iiii++)ao_type[iiii]='H';
            }
            if(s[sh_i].contr[0].l==6){
                type_shell='I';
                for(int iiii=3*ao_num; iiii<3*ao_num+84;iiii++)ao_type[iiii]='I';
            }
            if(s[sh_i].contr[0].l==7){
                type_shell='K';
                for(int iiii=3*ao_num; iiii<3*ao_num+108;iiii++)ao_type[iiii]='K';
            }
            for(k=0;k<s[sh_i].alpha.size();k++){
                fprintf(out," %3d   %c %4d%15.6f% 12.6f (% 10.6f) \n",
                        shell_num,type_shell,gto_num,
                        s[sh_i].alpha[k],s[sh_i].contr[0].coeff[k],
                        lib_coef[shell_num-1][k]);
                        
                gto_num++;
            }
            
            shell_num++;
            fprintf(out,"\n");
            ao_num+=s[sh_i].contr[0].size();
        }

    }
    fprintf(out," TOTAL NUMBER OF SHELLS              =  %3d\n",shell_num-1);
    fprintf(out," TOTAL NUMBER OF BASIS FUNCTIONS     =  %3d\n",n_ao);
    fprintf(out," NUMBER OF ELECTRONS                 =  %3d\n",n_el_full);
    fprintf(out," CHARGE OF MOLECULE                  =  %3d\n",mol_charge);
//     fprintf(out," STATE MULTIPLICITY                  =    %d\n");
//     fprintf(out," NUMBER OF OCCUPIED ORBITALS (ALPHA) =    %d\n");
//     fprintf(out," NUMBER OF OCCUPIED ORBITALS (BETA ) =    %d\n");
    fprintf(out,"\n");
    fprintf(out," FINAL ENERGY IS      0 AFTER  0 ITERATIONS\n");
    fprintf(out,"\n");
    fprintf(out,"          ------------\n");
    fprintf(out,"          EIGENVECTORS\n");
    fprintf(out,"          ------------\n");
    i=0;
//     exit(0);
    while((i+5)<=n_print_mo){
        fprintf(out,"\n");
        fprintf(out,"            ");
        for(k=1;k<6;k++)fprintf(out,"%11d",i+k);
        fprintf(out,"\n");
        fprintf(out,"                % 10.4f % 10.4f % 10.4f % 10.4f % 10.4f\n",orb_energy[i+0]
                                                                               ,orb_energy[i+1]
                                                                               ,orb_energy[i+2]
                                                                               ,orb_energy[i+3]
                                                                               ,orb_energy[i+4]);
        fprintf(out,"              ");
        S.print_rep(out,rep_num[i+0]);
        S.print_rep(out,rep_num[i+1]);
        S.print_rep(out,rep_num[i+2]);
        S.print_rep(out,rep_num[i+3]);
        S.print_rep(out,rep_num[i+4]);
        fprintf(out,"\n");        
        for(j=0;j<n_ao;j++){
            fprintf(out,"%5d  %c%c%3d%c%c%c",j+1,atom_names[atom_num[j]][0],
                                                 atom_names[atom_num[j]][1],
                                                 atom_num[j]+1,
                                                 ao_type[3*j  ],
                                                 ao_type[3*j+1],
                                                 ao_type[3*j+2]);
            for(k=0;k<5;k++)fprintf(out,"% 11.6f",MO_VEC_R[(i+k)*n_ao+/*new_ao_num[*/j/*]*/]);
            fprintf(out,"\n");    
        }
        i+=5;
    }
    if(i!=n_print_mo){
    fprintf(out,"\n");
    fprintf(out,"            ");
    for(k=i+1;k<=n_print_mo;k++)fprintf(out,"%11d",k);
    fprintf(out,"\n");
    fprintf(out,"               ");
    for(k=i+1;k<=n_print_mo;k++)fprintf(out," % 10.4f",orb_energy[k-1]);
    fprintf(out,"\n");
    fprintf(out,"              ");
    for(k=i+1;k<=n_print_mo;k++)S.print_rep(out,rep_num[k-1]);
    fprintf(out,"\n");
//     fprintf(out,"\n");
    for(j=0;j<n_ao;j++){
        fprintf(out,"%5d  %c%c%3d%c%c%c",j+1,atom_names[atom_num[j]][0],
                                                  atom_names[atom_num[j]][1],
                                                  atom_num[j]+1,
                                                  ao_type[3*j  ],
                                                  ao_type[3*j+1],
                                                  ao_type[3*j+2]);
        for(k=i;k<n_print_mo;k++)fprintf(out,"% 11.6f",MO_VEC_R[k*n_ao+/*new_ao_num[*/j/*]*/]);
        fprintf(out,"\n");
    }    
    }
    fprintf(out,"\n");
    fprintf(out,"\n");
    fprintf(out,"\n");
    
    delete[] atom_num   ;
    delete[] ao_type;
    
    
    fclose(out);
    return 0;
}

int molecule::GAMESS_geom_print(const char* out_name){
    
    FILE * out;
    out = fopen(out_name,"w");
    if(out==NULL){fprintf(out_stream,"couldn't open file %s to print structure\n",out_name);
                  return 1;}
    
    fprintf(out," GAMESS VERSIONS\n");
    fprintf(out," ATOM      ATOMIC                      COORDINATES (BOHR)\n");
    fprintf(out,"           CHARGE         X                   Y                   Z\n");
    for(int i=0;i<n_atoms;i++)
        fprintf(out,"%c%c          %.1f % 16.10f    % 16.10f    % 16.10f\n",
//         fprintf(out,"%c%c          %.1f     %.10f       %.10f       %.10f\n",
                    atom_names[i][0],atom_names[i][1],
                    nucl_charges_full[i],
                    atom_coord[3*i  ],atom_coord[3*i+1],atom_coord[3*i+2]);
    fprintf(out,"\n");
    fprintf(out,"\n");
    fprintf(out,"\n");
    fprintf(out,"\n");
    fclose(out);
    return 0;
}


int mol2_fprint(const char* out_name, molecule * a, molecule * b){
    
    int i;   
    FILE * out;
    out = fopen(out_name,"w");
    
    if(out==NULL){fprintf(out_stream,"couldn't open file %s to print structure\n",out_name);
                  return 1;}
    
    fprintf(out," GAMESS VERSIONS\n");
    fprintf(out," ATOM      ATOMIC                      COORDINATES (BOHR)\n");
    fprintf(out,"           CHARGE         X                   Y                   Z\n");
    for(i=0;i<a->n_atoms;i++)
    fprintf(out,"%c%c          %.1f % 16.10f    % 16.10f    % 16.10f\n",
//     fprintf(out,"%c%c          %.1f     %.10f       %.10f       %.10f\n",
                a->atom_names[i][0],a->atom_names[i][1],
                a->nucl_charges_full[i],
                a->atom_coord[3*i  ],a->atom_coord[3*i+1],a->atom_coord[3*i+2]);
    for(i=0;i<b->n_atoms;i++)
    fprintf(out,"%c%c          %.1f % 16.10f    % 16.10f    % 16.10f\n",
//     fprintf(out,"%c%c          %.1f     %.10f       %.10f       %.10f\n",
                b->atom_names[i][0],b->atom_names[i][1],
                b->nucl_charges_full[i],
                b->atom_coord[3*i  ],b->atom_coord[3*i+1],b->atom_coord[3*i+2]);
    fprintf(out," \n");
    fclose(out);
    return 0;
}

double molecule::center_of_mass(int xyz){
    double M=0.0;
    double Rc=0.0;
    int i;
    for(i=0;i<n_atoms;i++)M+=atom_mass[i];
//     fprintf(out_stream,"molecule mass = %f\n",M);
    for(i=0;i<n_atoms;i++)Rc+=atom_mass[i]*atom_coord[3*i+xyz];
    return Rc/M;
}

int molecule::inertion_tensor(double * T){
    
    
    set_zero_matr(T,9);
    double x,y,z,m;
    
    
    for(int i=0;i<n_atoms;i++){
        x=atom_coord[3*i+0];
        y=atom_coord[3*i+1];
        z=atom_coord[3*i+2];
        m=atom_mass[i];
        T[0]+=m*(y*y+z*z);
        T[1]-=m*x*y;
        T[2]-=m*x*z;
        T[3]-=m*x*y;
        T[4]+=m*(x*x+z*z);
        T[5]-=m*y*z;
        T[6]-=m*x*z;
        T[7]-=m*y*z;
        T[8]+=m*(x*x+y*y);
    }
   
//     PrintMatr(T,3,3,0);
    
    return 0;
}


// int molecule::move_to_mcs(int calc_rm, int inv){
//     
//     printf("ERROR: move to mcs is deprecated\n");
//     exit(0);
//     return 0;
//     
//     double x=center_of_mass(0);
//     double y=center_of_mass(1);
//     double z=center_of_mass(2);
//     
// //     printf("%f, %f, %f\n", center_of_mass(0), center_of_mass(1), center_of_mass(2));
//     
//     for(int i=0;i<n_atoms;i++){
//         atom_coord[3*i+0]-=x;
//         atom_coord[3*i+1]-=y;
//         atom_coord[3*i+2]-=z;
//     }
//     
// //     printf("%f, %f, %f\n", center_of_mass(0), center_of_mass(1), center_of_mass(2));
//     
//     double * T = mcs_rot_matr;
//     
//     if(calc_rm){
//         double *A = new double[3];
//         
//         inertion_tensor(T);
//         
//         lapack_diag(T,A,3);
//         
// //         PrintMatr(A,3,1,0);
// //         getchar();
//         
//         if(inv){
//             double tmp;
// //             tmp=T[0];T[0]=-T[6];T[6]=-tmp;
// //             tmp=T[1];T[1]=-T[7];T[7]=-tmp;
// //             tmp=T[2];T[2]=-T[8];T[8]=-tmp;
// //             
// //             T[3]=-T[3];
// //             T[4]=-T[4];
// //             T[5]=-T[5];
// //             
//             tmp=T[0];T[0]=-T[3];T[3]=-T[6];T[6]=tmp;
//             tmp=T[1];T[1]=-T[4];T[4]=-T[7];T[7]=tmp;
//             tmp=T[2];T[2]=-T[5];T[5]=-T[8];T[8]=tmp;
// 
//         }
//         
//         delete[] A;
//     }
//     
//     
//     for(int i=0;i<n_atoms;i++){
//         x=T[0]*atom_coord[3*i+0]+T[1]*atom_coord[3*i+1]+T[2]*atom_coord[3*i+2];
//         y=T[3]*atom_coord[3*i+0]+T[4]*atom_coord[3*i+1]+T[5]*atom_coord[3*i+2];
//         z=T[6]*atom_coord[3*i+0]+T[7]*atom_coord[3*i+1]+T[8]*atom_coord[3*i+2];
//         atom_coord[3*i+0]=x;//if(inv)atom_coord[3*i+0]= z;
//         atom_coord[3*i+1]=y;//if(inv)atom_coord[3*i+1]=-y;
//         atom_coord[3*i+2]=z;//if(inv)atom_coord[3*i+2]=-x;
//         
//     }
//     
//     return 0;
// }




double E_nuc_calc(molecule * a){
    
    int i,j;
    double V=0.0;
//     fprintf(out_stream,"n_atoms = %d\n", a->n_atoms);
    for(i=0;i<a->n_atoms;i++)
    for(j=i+1;j<a->n_atoms;j++)
        if((a->atom_is_ghost[i]*a->atom_is_ghost[j])==0)
            V+=a->nucl_charges_calc[i]*a->nucl_charges_calc[j]/dist(a->atom_coord[3*i],a->atom_coord[3*i+1],a->atom_coord[3*i+2],
                                                                    a->atom_coord[3*j],a->atom_coord[3*j+1],a->atom_coord[3*j+2],1);
//     fprintf(out_stream,"n_atoms = %d\n", a->n_atoms);
    
    return V;
}


double V_nuc_calc(molecule *a, molecule *b){
    
    int i,j;
    double V=0.0;
    if(a==b){fprintf(out_stream,"Warning equal arguments of V_nuc_calc!\nChanging to E_nuc_calc.\n");
        E_nuc_calc(a);
    }
    for(i=0;i<a->n_atoms;i++)
    for(j=0;j<b->n_atoms;j++){
        if((a->atom_is_ghost[i]*b->atom_is_ghost[j])==0)
            V+=a->nucl_charges_calc[i]*b->nucl_charges_calc[j]/dist(a->atom_coord[3*i],a->atom_coord[3*i+1],a->atom_coord[3*i+2],
                                                                    b->atom_coord[3*j],b->atom_coord[3*j+1],b->atom_coord[3*j+2],1);
    }
    return V;
}

double D_nuc_calc(molecule * a, int xyz){
    
    int i;
    double D=0.0;
//     fprintf(out_stream,"n_atoms = %d\n", a->n_atoms);
    for(i=0;i<a->n_atoms;i++)
        D+=a->nucl_charges_calc[i]*a->atom_coord[3*i+xyz];
//     fprintf(out_stream,"n_atoms = %d\n", a->n_atoms);
    
    return D;
}


# ifdef LIBINT
int molecule::add_libint_atoms(vector<Atom> * atoms){
//     fprintf(out_stream,"atoms %d\n",(*atoms).size());
    int n0=(*atoms).size();
    atoms->resize(n0+n_atoms);
    for(int i=0;i<n_atoms;i++){
        (*atoms)[i+n0].atomic_number = int(nucl_charges_calc[i]);
        (*atoms)[i+n0].x = atom_coord[3*i  ];
        (*atoms)[i+n0].y = atom_coord[3*i+1];
        (*atoms)[i+n0].z = atom_coord[3*i+2];
//         fprintf(out_stream," c = %d\n",(*atoms)[i+n0].atomic_number);
    }
    
    return 0;
}

int molecule::MO_libint_format(){
        
    int j,k;
    double c;

    for(j=0; j<n_ro;j++){
        k=0;
        for(auto &AO: read_s){
            if(AO.contr[0].l==0){k++;continue;}
            if(AO.contr[0].l==1){k+=3;continue;}
            if(AO.contr[0].l==2){
                // XX-XX
                // YY-XY
                // ZZ-XZ
                // XY-YY
                // XZ-YZ
                // YZ-ZZ
                // 0=0
                // 1<->3
                // 2<-4<-5<-2
                c                 =MO_VEC_R[j*n_ro+k+1];
                MO_VEC_R[j*n_ro+k+1]=MO_VEC_R[j*n_ro+k+3]*sqrt(3.0);
                MO_VEC_R[j*n_ro+k+3]=c;
                c                 =MO_VEC_R[j*n_ro+k+2];
                MO_VEC_R[j*n_ro+k+2]=MO_VEC_R[j*n_ro+k+4]*sqrt(3.0);
                MO_VEC_R[j*n_ro+k+4]=MO_VEC_R[j*n_ro+k+5]*sqrt(3.0);
                MO_VEC_R[j*n_ro+k+5]=c;
                k+=6;
                continue;
            }
            if(AO.contr[0].l==3){
                // XXX-XXX 0
                // YYY-XXY 1
                // ZZZ-XXZ 2
                // XXY-XYY(YYX) 3
                // XXZ-XYZ 4
                // YYX-XZZ(ZZX) 5
                // YYZ-YYY 6
                // ZZX-YYZ 7
                // ZZY-YZZ 8
                // XYZ-ZZZ 9
                // 0=0
                // 1<-3<-5<-7<-6<-1
                // 2<-4<-9<-2
                // 8=8
                c                 =MO_VEC_R[j*n_ro+k+1]          ;
                MO_VEC_R[j*n_ro+k+1]=MO_VEC_R[j*n_ro+k+3]*sqrt( 5.0);
                MO_VEC_R[j*n_ro+k+3]=MO_VEC_R[j*n_ro+k+5]*sqrt( 5.0);
                MO_VEC_R[j*n_ro+k+5]=MO_VEC_R[j*n_ro+k+7]*sqrt( 5.0);
                MO_VEC_R[j*n_ro+k+7]=MO_VEC_R[j*n_ro+k+6]*sqrt( 5.0);
                MO_VEC_R[j*n_ro+k+6]=c;
                c                 =MO_VEC_R[j*n_ro+k+2];
                MO_VEC_R[j*n_ro+k+2]=MO_VEC_R[j*n_ro+k+4]*sqrt( 5.0);
                MO_VEC_R[j*n_ro+k+4]=MO_VEC_R[j*n_ro+k+9]*sqrt(15.0);
                MO_VEC_R[j*n_ro+k+9]=c;
                MO_VEC_R[j*n_ro+k+8]=MO_VEC_R[j*n_ro+k+8]*sqrt( 5.0);
                k+=10;
                continue;
            }
            if(AO.contr[0].l==4){
                // XXXX-XXXX 0
                // YYYY-XXXY 1
                // ZZZZ-XXXZ 2
                // XXXY-XXYY 3
                // XXXZ-XXYZ 4
                // YYYX-XXZZ 5
                // YYYZ-XYYY (YYYX) 6
                // ZZZX-XYYZ (YYXZ) 7
                // ZZZY-XYZZ (ZZXY) 8
                // XXYY-XZZZ (ZZZX) 9
                // XXZZ YYYY 10
                // YYZZ YYYZ 11
                // XXYZ YYZZ 12
                // YYXZ YZZZ (ZZZY) 13
                // ZZXY ZZZZ 14
                
                // 0=0
                // 1<-3<-9<-7<-13<-8<-14<-2<-4<-12<-11<-6<-5<-10<-1
                // 
                
                
                c                  =MO_VEC_R[j*n_ro+k+1 ];
                MO_VEC_R[j*n_ro+k+1 ]=MO_VEC_R[j*n_ro+k+3 ]*sqrt( 7.0);
                MO_VEC_R[j*n_ro+k+3 ]=MO_VEC_R[j*n_ro+k+9 ]*sqrt(35.0/3.0);
                MO_VEC_R[j*n_ro+k+9 ]=MO_VEC_R[j*n_ro+k+7 ]*sqrt( 7.0);
                MO_VEC_R[j*n_ro+k+7 ]=MO_VEC_R[j*n_ro+k+13]*sqrt(35.0);
                MO_VEC_R[j*n_ro+k+13]=MO_VEC_R[j*n_ro+k+8 ]*sqrt( 7.0);
                MO_VEC_R[j*n_ro+k+8 ]=MO_VEC_R[j*n_ro+k+14]*sqrt(35.0);
                MO_VEC_R[j*n_ro+k+14]=MO_VEC_R[j*n_ro+k+2 ];
                MO_VEC_R[j*n_ro+k+2 ]=MO_VEC_R[j*n_ro+k+4 ]*sqrt( 7.0);
                MO_VEC_R[j*n_ro+k+4 ]=MO_VEC_R[j*n_ro+k+12]*sqrt(35.0);
                MO_VEC_R[j*n_ro+k+12]=MO_VEC_R[j*n_ro+k+11]*sqrt(35.0/3.0);
                MO_VEC_R[j*n_ro+k+11]=MO_VEC_R[j*n_ro+k+6 ]*sqrt( 7.0);
                MO_VEC_R[j*n_ro+k+6 ]=MO_VEC_R[j*n_ro+k+5 ]*sqrt( 7.0);
                MO_VEC_R[j*n_ro+k+5 ]=MO_VEC_R[j*n_ro+k+10]*sqrt(35.0/3.0);
                MO_VEC_R[j*n_ro+k+10]=c;
                k+=15;
                continue;
            }
            if(AO.contr[0].l==5){
                // XXXXX-XXXXX   0             +
                // YYYYY-XXXXY   1             +
                // ZZZZZ-XXXXZ   2             +
                // XXXXY-XXXYY   3             +
                // XXXXZ-XXXYZ   4             +
                // YYYYX-XXXZZ   5             +
                // YYYYZ-XXYYY   6  (YYYXX)    +
                // ZZZZX-XXYYZ   7             +
                // ZZZZY-XXYZZ   8  (XXZZY)    +
                // XXXYY-XXZZZ   9  (ZZZXX)    +
                // XXXZZ-XYYYY   10 (YYYYX)    +
                // YYYXX-XYYYZ   11 (YYYXZ)    +
                // YYYZZ-XYYZZ   12 (YYZZX)    +
                // ZZZXX-XYZZZ   13 (ZZZXY)
                // ZZZYY-XZZZZ   14 (ZZZZX)    +
                // XXXYZ-YYYYY   15            +
                // YYYXZ-YYYYZ   16            +
                // ZZZXY-YYYZZ   17            +
                // XXYYZ-YYZZZ   18 (ZZZYY)    +
                // XXZZY-YZZZZ   19 (ZZZZY)    +
                // YYZZX-ZZZZZ   20            +
                
                // 0=0
                // 1<-3<-9<-13<-17<-12<-20<-2<-4<-15<-1
                // 5<->10
                // 6<-11<-16<-6
                // 7<-18<-14<-7
                // 8<->19
                
                
                
                c                    =MO_VEC_R[j*n_ro+k+3 ]*sqrt(  9.0);
                MO_VEC_R[j*n_ro+k+3 ]=MO_VEC_R[j*n_ro+k+9 ]*sqrt( 21.0);
                MO_VEC_R[j*n_ro+k+9 ]=MO_VEC_R[j*n_ro+k+13]*sqrt( 21.0);
                MO_VEC_R[j*n_ro+k+13]=MO_VEC_R[j*n_ro+k+17]*sqrt( 63.0);
                MO_VEC_R[j*n_ro+k+17]=MO_VEC_R[j*n_ro+k+12]*sqrt( 21.0);
                MO_VEC_R[j*n_ro+k+12]=MO_VEC_R[j*n_ro+k+20]*sqrt(105.0);
                MO_VEC_R[j*n_ro+k+20]=MO_VEC_R[j*n_ro+k+2 ];
                MO_VEC_R[j*n_ro+k+2 ]=MO_VEC_R[j*n_ro+k+4 ]*sqrt(  9.0);
                MO_VEC_R[j*n_ro+k+4 ]=MO_VEC_R[j*n_ro+k+15]*sqrt( 63.0);
                MO_VEC_R[j*n_ro+k+15]=MO_VEC_R[j*n_ro+k+1 ];
                MO_VEC_R[j*n_ro+k+1 ]=c                  ;
                c                  =MO_VEC_R[j*n_ro+k+10]*sqrt( 21.0);
                MO_VEC_R[j*n_ro+k+10]=MO_VEC_R[j*n_ro+k+5 ]*sqrt(  9.0);
                MO_VEC_R[j*n_ro+k+5 ]=c                  ;
                c                  =MO_VEC_R[j*n_ro+k+11]*sqrt( 21.0);
                MO_VEC_R[j*n_ro+k+11]=MO_VEC_R[j*n_ro+k+16]*sqrt( 63.0);
                MO_VEC_R[j*n_ro+k+16]=MO_VEC_R[j*n_ro+k+6 ]*sqrt(  9.0);
                MO_VEC_R[j*n_ro+k+6 ]=c                  ;
                c                  =MO_VEC_R[j*n_ro+k+18]*sqrt(105.0);
                MO_VEC_R[j*n_ro+k+18]=MO_VEC_R[j*n_ro+k+14]*sqrt( 21.0);
                MO_VEC_R[j*n_ro+k+14]=MO_VEC_R[j*n_ro+k+7 ]*sqrt(  9.0);
                MO_VEC_R[j*n_ro+k+7 ]=c                  ;
                c                  =MO_VEC_R[j*n_ro+k+19]*sqrt(105.0);
                MO_VEC_R[j*n_ro+k+19]=MO_VEC_R[j*n_ro+k+8 ]*sqrt(  9.0);
                MO_VEC_R[j*n_ro+k+8 ]=c                  ;
                k+=21;
                continue;
            }
            if(AO.contr[0].l==6){
                // XXXXXX-XXXXXX   0            +
                // YYYYYY-XXXXXY   1            +
                // ZZZZZZ-XXXXXZ   2            +
                // XXXXXY-XXXXYY   3            +
                // XXXXXZ-XXXXYZ   4            +
                // YYYYYX-XXXXZZ   5            +
                // YYYYYZ-XXXYYY   6            +
                // ZZZZZX-XXXYYZ   7            +
                // ZZZZZY-XXXYZZ   8  (XXXZZY)  +
                // XXXXYY-XXXZZZ   9            +
                // XXXXZZ-XXYYYY   10 (YYYYXX)  +
                // YYYYXX-XXYYYZ   11 (YYYXXZ)  +
                // YYYYZZ-XXYYZZ   12           +
                // ZZZZXX-XXYZZZ   13 (ZZZXXY)  +
                // ZZZZYY-XXZZZZ   14 (ZZZZXX)  +
                // XXXXYZ-XYYYYY   15 (YYYYYX)  +
                // YYYYXZ-XYYYYZ   16 (YYYYXZ)  +
                // ZZZZXY-XYYYZZ   17 (YYYZZX)  +
                // XXXYYY-XYYZZZ   18 (ZZZYYX)  +
                // XXXZZZ-XYZZZZ   19 (ZZZZXY)  +
                // YYYZZZ-XZZZZZ   20 (ZZZZZX)  +
                // XXXYYZ-YYYYYY   21           +
                // XXXZZY-YYYYYZ   22           +
                // YYYXXZ-YYYYZZ   23           +
                // YYYZZX-YYYZZZ   24           +
                // ZZZXXY-YYZZZZ   25 (ZZZZYY)  +
                // ZZZYYX-YZZZZZ   26 (ZZZZZY)  +
                // XXYYZZ-ZZZZZZ   27           +
                
                // 0=0
                // 1<-3<-9<-19<-17<-24<-20<-7<-21<-1
                // 2<-4<-15<-5<-10<-11<-23<-12<-27<-2
                // 6<-18<-26<-8<-22<-6
                // 13<-25<-14<-13
                // 16<-16
                
                
                MO_VEC_R[j*n_ao+k   ]=MO_VEC_R[j*n_ao+k   ];//no
                c                    =MO_VEC_R[j*n_ao+k+3 ]*sqrt(11.0);
                MO_VEC_R[j*n_ao+k+3 ]=MO_VEC_R[j*n_ao+k+9 ]*sqrt(33.0);
                MO_VEC_R[j*n_ao+k+9 ]=MO_VEC_R[j*n_ao+k+19]*sqrt(231.0/5.0);
                MO_VEC_R[j*n_ao+k+19]=MO_VEC_R[j*n_ao+k+17]*sqrt(99.0);
                MO_VEC_R[j*n_ao+k+17]=MO_VEC_R[j*n_ao+k+24]*sqrt(231.0);
                MO_VEC_R[j*n_ao+k+24]=MO_VEC_R[j*n_ao+k+20]*sqrt(231.0/5.0);
                MO_VEC_R[j*n_ao+k+20]=MO_VEC_R[j*n_ao+k+7 ]*sqrt(11.0);
                MO_VEC_R[j*n_ao+k+7 ]=MO_VEC_R[j*n_ao+k+21]*sqrt(231.0);
                MO_VEC_R[j*n_ao+k+21]=MO_VEC_R[j*n_ao+k+1 ];//no
                MO_VEC_R[j*n_ao+k+1 ]=c;
                // 2<-4<-15<-5<-10<-11<-23<-12<-27<-2
                c                    =MO_VEC_R[j*n_ao+k+4 ]*sqrt(11.0);
                MO_VEC_R[j*n_ao+k+4 ]=MO_VEC_R[j*n_ao+k+15]*sqrt(99.0);
                MO_VEC_R[j*n_ao+k+15]=MO_VEC_R[j*n_ao+k+5 ]*sqrt(11.0);
                MO_VEC_R[j*n_ao+k+5 ]=MO_VEC_R[j*n_ao+k+10]*sqrt(33.0);
                MO_VEC_R[j*n_ao+k+10]=MO_VEC_R[j*n_ao+k+11]*sqrt(33.0);
                MO_VEC_R[j*n_ao+k+11]=MO_VEC_R[j*n_ao+k+23]*sqrt(231.0);
                MO_VEC_R[j*n_ao+k+23]=MO_VEC_R[j*n_ao+k+12]*sqrt(33.0);
                MO_VEC_R[j*n_ao+k+12]=MO_VEC_R[j*n_ao+k+27]*sqrt(385.0);
                MO_VEC_R[j*n_ao+k+27]=MO_VEC_R[j*n_ao+k+2 ];//no
                MO_VEC_R[j*n_ao+k+2 ]=c                  ;
                // 6<-18<-26<-8<-22<-6
                c                    =MO_VEC_R[j*n_ao+k+18]*sqrt(231.0/5.0);
                MO_VEC_R[j*n_ao+k+18]=MO_VEC_R[j*n_ao+k+26]*sqrt(231.0);
                MO_VEC_R[j*n_ao+k+26]=MO_VEC_R[j*n_ao+k+8 ]*sqrt(11.0);
                MO_VEC_R[j*n_ao+k+8 ]=MO_VEC_R[j*n_ao+k+22]*sqrt(231.0);
                MO_VEC_R[j*n_ao+k+22]=MO_VEC_R[j*n_ao+k+6 ]*sqrt(11.0);
                MO_VEC_R[j*n_ao+k+6 ]=c;
                // 13<-25<-14<-13
                c                    =MO_VEC_R[j*n_ao+k+25]*sqrt(231.0);
                MO_VEC_R[j*n_ao+k+25]=MO_VEC_R[j*n_ao+k+14]*sqrt(33.0);
                MO_VEC_R[j*n_ao+k+14]=MO_VEC_R[j*n_ao+k+13]*sqrt(33.0);
                MO_VEC_R[j*n_ao+k+13]=c                 ;
                MO_VEC_R[j*n_ao+k+16]=MO_VEC_R[j*n_ao+k+16]*sqrt(99.0);
                
                k+=28;
                continue;
            }
            
            if(AO.contr[0].l>6){
                fprintf(out_stream,"ERROR: MO_libint_format is not realized for orbital higher than I\n");
                exit(1);
            }
            
        }
    }
    
    return 0;
}

int molecule::MO_gamess_format(){
    
    int j,k;
    double c;

    for(j=0; j<n_mo;j++){
        k=0;
        for(auto &AO: s){
            if(AO.contr[0].l==0){
                MO_VEC_R[j*n_ao+k]=MO_VEC[j*n_ao+k];
                k++;
                continue;
                
            }
            if(AO.contr[0].l==1){
                MO_VEC_R[j*n_ao+k  ]=MO_VEC[j*n_ao+k  ];
                MO_VEC_R[j*n_ao+k+1]=MO_VEC[j*n_ao+k+1];
                MO_VEC_R[j*n_ao+k+2]=MO_VEC[j*n_ao+k+2];
                k+=3;
                continue;
            }
            if(AO.contr[0].l==2){
                // XX-XX
                // YY-XY
                // ZZ-XZ
                // XY-YY
                // XZ-YZ
                // YZ-ZZ
                // 0=0
                // 1<->3
                // 2<-5<-4<-2
                MO_VEC_R[j*n_ao+k  ]=MO_VEC[j*n_ao+k  ]           ;
                c                   =MO_VEC[j*n_ao+k+3];
                MO_VEC_R[j*n_ao+k+3]=MO_VEC[j*n_ao+k+1]/sqrt(3.0);
                MO_VEC_R[j*n_ao+k+1]=c;
                c                   =MO_VEC[j*n_ao+k+5];
                MO_VEC_R[j*n_ao+k+5]=MO_VEC[j*n_ao+k+4]/sqrt(3.0);
                MO_VEC_R[j*n_ao+k+4]=MO_VEC[j*n_ao+k+2]/sqrt(3.0);
                MO_VEC_R[j*n_ao+k+2]=c;
                k+=6;
                continue;
            }
            if(AO.contr[0].l==3){
                // XXX-XXX 0
                // YYY-XXY 1
                // ZZZ-XXZ 2
                // XXY-XYY(YYX) 3
                // XXZ-XYZ 4
                // YYX-XZZ(ZZX) 5
                // YYZ-YYY 6
                // ZZX-YYZ 7
                // ZZY-YZZ 8
                // XYZ-ZZZ 9
                // 0=0
                // 1<-6<-7<-5<-3<-1
                // 2<-9<-4<-2
                // 8=8
                
                MO_VEC_R[j*n_ao+k  ]=MO_VEC[j*n_ao+k  ]           ;
                c                   =MO_VEC[j*n_ao+k+1]/sqrt( 5.0);
                MO_VEC_R[j*n_ao+k+1]=MO_VEC[j*n_ao+k+6]           ;
                MO_VEC_R[j*n_ao+k+6]=MO_VEC[j*n_ao+k+7]/sqrt( 5.0);
                MO_VEC_R[j*n_ao+k+7]=MO_VEC[j*n_ao+k+5]/sqrt( 5.0);
                MO_VEC_R[j*n_ao+k+5]=MO_VEC[j*n_ao+k+3]/sqrt( 5.0);
                MO_VEC_R[j*n_ao+k+3]=c;
                c                   =MO_VEC[j*n_ao+k+2]/sqrt( 5.0);
                MO_VEC_R[j*n_ao+k+2]=MO_VEC[j*n_ao+k+9]           ;
                MO_VEC_R[j*n_ao+k+9]=MO_VEC[j*n_ao+k+4]/sqrt(15.0);
                MO_VEC_R[j*n_ao+k+4]=c;
                MO_VEC_R[j*n_ao+k+8]=MO_VEC[j*n_ao+k+8]/sqrt( 5.0);
                k+=10;
                continue;
            }
            if(AO.contr[0].l==4){
                // XXXX-XXXX 0
                // YYYY-XXXY 1
                // ZZZZ-XXXZ 2
                // XXXY-XXYY 3
                // XXXZ-XXYZ 4
                // YYYX-XXZZ 5
                // YYYZ-XYYY (YYYX) 6
                // ZZZX-XYYZ (YYXZ) 7
                // ZZZY-XYZZ (ZZXY) 8
                // XXYY-XZZZ (ZZZX) 9
                // XXZZ YYYY 10
                // YYZZ YYYZ 11
                // XXYZ YYZZ 12
                // YYXZ YZZZ (ZZZY) 13
                // ZZXY ZZZZ 14
                
                // 0=0
                // 1<-10<-5<-6 <-11<-12<-4<-2<-14<-8<-13<-7<-9<-3<-1
                // 
                
                MO_VEC_R[j*n_ao+k   ]=MO_VEC[j*n_ao+k   ]           ;
                c                    =MO_VEC[j*n_ao+k+10];
                MO_VEC_R[j*n_ao+k+10]=MO_VEC[j*n_ao+k+5 ]/sqrt(35.0/3.0);
                MO_VEC_R[j*n_ao+k+5 ]=MO_VEC[j*n_ao+k+6 ]/sqrt( 7.0);
                MO_VEC_R[j*n_ao+k+6 ]=MO_VEC[j*n_ao+k+11]/sqrt( 7.0);
                MO_VEC_R[j*n_ao+k+11]=MO_VEC[j*n_ao+k+12]/sqrt(35.0/3.0);
                MO_VEC_R[j*n_ao+k+12]=MO_VEC[j*n_ao+k+4 ]/sqrt(35.0);
                MO_VEC_R[j*n_ao+k+4 ]=MO_VEC[j*n_ao+k+2 ]/sqrt( 7.0);
                MO_VEC_R[j*n_ao+k+2 ]=MO_VEC[j*n_ao+k+14];
                MO_VEC_R[j*n_ao+k+14]=MO_VEC[j*n_ao+k+8 ]/sqrt(35.0);
                MO_VEC_R[j*n_ao+k+8 ]=MO_VEC[j*n_ao+k+13]/sqrt( 7.0);
                MO_VEC_R[j*n_ao+k+13]=MO_VEC[j*n_ao+k+7 ]/sqrt(35.0);
                MO_VEC_R[j*n_ao+k+7 ]=MO_VEC[j*n_ao+k+9 ]/sqrt( 7.0);
                MO_VEC_R[j*n_ao+k+9 ]=MO_VEC[j*n_ao+k+3 ]/sqrt(35.0/3.0);
                MO_VEC_R[j*n_ao+k+3 ]=MO_VEC[j*n_ao+k+1 ]/sqrt( 7.0);
                MO_VEC_R[j*n_ao+k+1 ]=c;
                k+=15;
                continue;
            }
            if(AO.contr[0].l==5){
                // XXXXX-XXXXX   0             +
                // YYYYY-XXXXY   1             +
                // ZZZZZ-XXXXZ   2             +
                // XXXXY-XXXYY   3             +
                // XXXXZ-XXXYZ   4             +
                // YYYYX-XXXZZ   5             +
                // YYYYZ-XXYYY   6  (YYYXX)    +
                // ZZZZX-XXYYZ   7             +
                // ZZZZY-XXYZZ   8  (XXZZY)    +
                // XXXYY-XXZZZ   9  (ZZZXX)    +
                // XXXZZ-XYYYY   10 (YYYYX)    +
                // YYYXX-XYYYZ   11 (YYYXZ)    +
                // YYYZZ-XYYZZ   12 (YYZZX)    +
                // ZZZXX-XYZZZ   13 (ZZZXY)
                // ZZZYY-XZZZZ   14 (ZZZZX)    +
                // XXXYZ-YYYYY   15            +
                // YYYXZ-YYYYZ   16            +
                // ZZZXY-YYYZZ   17            +
                // XXYYZ-YYZZZ   18 (ZZZYY)    +
                // XXZZY-YZZZZ   19 (ZZZZY)    +
                // YYZZX-ZZZZZ   20            +
                
                // 0=0
                // 1<-15<-4<-2<-20<-12<-17<-13<-9<-3<-1
                // 5<->10
                // 6<-16<-11<-6
                // 7<-14<-18<-7
                // 8<->19
                
                
                MO_VEC_R[j*n_ao+k   ]=MO_VEC[j*n_ao+k   ]           ;
                c                    =MO_VEC[j*n_ao+k+15];//no
                MO_VEC_R[j*n_ao+k+15]=MO_VEC[j*n_ao+k+4 ]/sqrt(63.0);
                MO_VEC_R[j*n_ao+k+4 ]=MO_VEC[j*n_ao+k+2 ]/sqrt( 9.0);
                MO_VEC_R[j*n_ao+k+2 ]=MO_VEC[j*n_ao+k+20];//no
                MO_VEC_R[j*n_ao+k+20]=MO_VEC[j*n_ao+k+12]/sqrt(105.0);
                MO_VEC_R[j*n_ao+k+12]=MO_VEC[j*n_ao+k+17]/sqrt( 21.0);
                MO_VEC_R[j*n_ao+k+17]=MO_VEC[j*n_ao+k+13]/sqrt( 63.0);
                MO_VEC_R[j*n_ao+k+13]=MO_VEC[j*n_ao+k+9 ]/sqrt( 21.0);
                MO_VEC_R[j*n_ao+k+9 ]=MO_VEC[j*n_ao+k+3 ]/sqrt( 21.0);
                MO_VEC_R[j*n_ao+k+3 ]=MO_VEC[j*n_ao+k+1 ]/sqrt(  9.0);
                MO_VEC_R[j*n_ao+k+1 ]=c                  ;
                c                    =MO_VEC[j*n_ao+k+10]/sqrt(  9.0);
                MO_VEC_R[j*n_ao+k+10]=MO_VEC[j*n_ao+k+5 ]/sqrt( 21.0);
                MO_VEC_R[j*n_ao+k+5 ]=c                  ;
                c                    =MO_VEC[j*n_ao+k+16]/sqrt(  9.0);
                MO_VEC_R[j*n_ao+k+16]=MO_VEC[j*n_ao+k+11]/sqrt( 63.0);
                MO_VEC_R[j*n_ao+k+11]=MO_VEC[j*n_ao+k+6 ]/sqrt( 21.0);
                MO_VEC_R[j*n_ao+k+6 ]=c                  ;
                c                    =MO_VEC[j*n_ao+k+14]/sqrt(  9.0);
                MO_VEC_R[j*n_ao+k+14]=MO_VEC[j*n_ao+k+18]/sqrt( 21.0);
                MO_VEC_R[j*n_ao+k+18]=MO_VEC[j*n_ao+k+7 ]/sqrt(105.0);
                MO_VEC_R[j*n_ao+k+7 ]=c                  ;
                c                    =MO_VEC[j*n_ao+k+19]/sqrt(  9.0);
                MO_VEC_R[j*n_ao+k+19]=MO_VEC[j*n_ao+k+8 ]/sqrt(105.0);
                MO_VEC_R[j*n_ao+k+8 ]=c                  ;
                k+=21;
                continue;
            }
            if(AO.contr[0].l==6){
                // XXXXXX-XXXXXX   0            +
                // YYYYYY-XXXXXY   1            +
                // ZZZZZZ-XXXXXZ   2            +
                // XXXXXY-XXXXYY   3            +
                // XXXXXZ-XXXXYZ   4            +
                // YYYYYX-XXXXZZ   5            +
                // YYYYYZ-XXXYYY   6            +
                // ZZZZZX-XXXYYZ   7            +
                // ZZZZZY-XXXYZZ   8  (XXXZZY)  +
                // XXXXYY-XXXZZZ   9            +
                // XXXXZZ-XXYYYY   10 (YYYYXX)  +
                // YYYYXX-XXYYYZ   11 (YYYXXZ)  +
                // YYYYZZ-XXYYZZ   12           +
                // ZZZZXX-XXYZZZ   13 (ZZZXXY)  +
                // ZZZZYY-XXZZZZ   14 (ZZZZXX)  +
                // XXXXYZ-XYYYYY   15 (YYYYYX)  +
                // YYYYXZ-XYYYYZ   16 (YYYYXZ)  +
                // ZZZZXY-XYYYZZ   17 (YYYZZX)  +
                // XXXYYY-XYYZZZ   18 (ZZZYYX)  +
                // XXXZZZ-XYZZZZ   19 (ZZZZXY)  +
                // YYYZZZ-XZZZZZ   20 (ZZZZZX)  +
                // XXXYYZ-YYYYYY   21           +
                // XXXZZY-YYYYYZ   22           +
                // YYYXXZ-YYYYZZ   23           +
                // YYYZZX-YYYZZZ   24           +
                // ZZZXXY-YYZZZZ   25 (ZZZZYY)  +
                // ZZZYYX-YZZZZZ   26 (ZZZZZY)  +
                // XXYYZZ-ZZZZZZ   27           +
                
                // 0=0
                // 1<-21<-7<-20<-24<-17<-19<-9<-3<-1
                // 2<-27<-12<-23<-11<-10<-5<-15<-4<-2
                // 6<-22<-8<-26<-18<-6
                // 13<-14<-25<-13
                // 16<-16
                
                
                MO_VEC_R[j*n_ao+k   ]=MO_VEC[j*n_ao+k   ];//no
                c                    =MO_VEC[j*n_ao+k+21];//no
                MO_VEC_R[j*n_ao+k+21]=MO_VEC[j*n_ao+k+7 ]/sqrt(231.0);
                MO_VEC_R[j*n_ao+k+7 ]=MO_VEC[j*n_ao+k+20]/sqrt(11.0);
                MO_VEC_R[j*n_ao+k+20]=MO_VEC[j*n_ao+k+24]/sqrt(231.0/5.0);
                MO_VEC_R[j*n_ao+k+24]=MO_VEC[j*n_ao+k+17]/sqrt(231.0);
                MO_VEC_R[j*n_ao+k+17]=MO_VEC[j*n_ao+k+19]/sqrt(99.0);
                MO_VEC_R[j*n_ao+k+19]=MO_VEC[j*n_ao+k+9 ]/sqrt(231.0/5.0);
                MO_VEC_R[j*n_ao+k+9 ]=MO_VEC[j*n_ao+k+3 ]/sqrt(33.0);
                MO_VEC_R[j*n_ao+k+3 ]=MO_VEC[j*n_ao+k+1 ]/sqrt(11.0);
                MO_VEC_R[j*n_ao+k+1 ]=c;
                c                    =MO_VEC[j*n_ao+k+27];//no
                MO_VEC_R[j*n_ao+k+27]=MO_VEC[j*n_ao+k+12]/sqrt(385.0);
                MO_VEC_R[j*n_ao+k+12]=MO_VEC[j*n_ao+k+23]/sqrt(33.0);
                MO_VEC_R[j*n_ao+k+23]=MO_VEC[j*n_ao+k+11]/sqrt(231.0);
                MO_VEC_R[j*n_ao+k+11]=MO_VEC[j*n_ao+k+10]/sqrt(33.0);
                MO_VEC_R[j*n_ao+k+10]=MO_VEC[j*n_ao+k+5 ]/sqrt(33.0);
                MO_VEC_R[j*n_ao+k+5 ]=MO_VEC[j*n_ao+k+15]/sqrt(11.0);
                MO_VEC_R[j*n_ao+k+15]=MO_VEC[j*n_ao+k+4 ]/sqrt(99.0);
                MO_VEC_R[j*n_ao+k+4 ]=MO_VEC[j*n_ao+k+2 ]/sqrt(11.0);
                MO_VEC_R[j*n_ao+k+2 ]=c                  ;
                c                    =MO_VEC[j*n_ao+k+22]/sqrt(11.0);
                MO_VEC_R[j*n_ao+k+22]=MO_VEC[j*n_ao+k+8 ]/sqrt(231.0);
                MO_VEC_R[j*n_ao+k+8 ]=MO_VEC[j*n_ao+k+26]/sqrt(11.0);
                MO_VEC_R[j*n_ao+k+26]=MO_VEC[j*n_ao+k+18]/sqrt(231.0);
                MO_VEC_R[j*n_ao+k+18]=MO_VEC[j*n_ao+k+6 ]/sqrt(231.0/5.0);
                MO_VEC_R[j*n_ao+k+6 ]=c;
                c                    =MO_VEC[j*n_ao+k+14]/sqrt(33.0);
                MO_VEC_R[j*n_ao+k+14]=MO_VEC[j*n_ao+k+25]/sqrt(33.0);
                MO_VEC_R[j*n_ao+k+25]=MO_VEC[j*n_ao+k+13]/sqrt(231.0);
                MO_VEC_R[j*n_ao+k+13]=c;
                MO_VEC_R[j*n_ao+k+16]=MO_VEC[j*n_ao+k+16]/sqrt(99.0);
                
                k+=28;
                continue;
            }
            if(AO.contr[0].l>6){
                fprintf(out_stream,"ERROR: MO_libint_back_reordr is not realized for orbital higher than I\n");
                exit(1);
            }
        }
    }
    
    return 0;
}


vector<pair<double,array<double, 3>>> molecule::libint_point_charges(){
          
    vector<pair<double, array<double, 3>>> q(n_atoms);
    for (int i=0; i<n_atoms; i++) {
      q.emplace_back(static_cast<double>(nucl_charges_calc[i]),
                     array<double, 3>{{atom_coord[3*i], atom_coord[3*i+1], atom_coord[3*i+2]}});
    }
    return q;
}

#endif




int molecule::efrag_read(){
        
    FILE *in;
    FILE *at;
    char * line = new char[BUF_LINE_LENGTH];
    char * at_name = new char[BUF_LINE_LENGTH];
    sprintf(at_name,"%s/atoms.dat\0",NOPT_LIB);
    
    
    int line_num=0;
    int start;
    
    
    in = fopen (source_name,"r");
    at = fopen (    at_name,"r");
    
    fgets(line,BUF_LINE_LENGTH,in),line_num++;
    while((key_word_comp(line,geom_dim_unit)==0)&&(!feof(in))){fgets(line,BUF_LINE_LENGTH,in),line_num++;}
    
    if(key_word_comp(line,geom_dim_unit)==0){
        fprintf(out_stream,"\n");
        fprintf(out_stream,"WARNING: units are not specified in %s\n",source_name);
        fprintf(out_stream,"         using default UNITS=BOHR\n\n");
        au_coef=1.0;
    }
    else{
        if(key_word_comp(line,geom_dim_au)) au_coef=1.0;
        if(key_word_comp(line,geom_dim_nm)) au_coef=18.89725989;
        if(key_word_comp(line,geom_dim_ang))au_coef=1.889725989;
    }
    
    fseek(in,0,SEEK_SET);
    line_num=0;
    
    smart_fgets(line,&line_num,in,source_name, "reading file");
    
    while((key_word_comp(line,geom_group_start)==0)&&(!feof(in)))smart_fgets(line,&line_num,in,source_name, "looking for molecular geometry (keywords: xyz, geom, geometry)");
    
    smart_fgets(line,&line_num,in,source_name,"reading number of atoms");
    
    n_atoms=atoi(line);
    n_el_full = 0;
    n_el_calc = 0;
    
    if(n_atoms<1){
        fprintf(out_stream,"ERROR: line %d in %s\n", line_num, source_name);
        fprintf(out_stream,"       number of atoms is illegal n=%d\n",n_atoms);
        exit(1);
    }
    atom_coord = new double[n_atoms*3];
    nucl_charges_full = new double[n_atoms];
    nucl_charges_calc = new double[n_atoms];
    atom_names = new char*[n_atoms];
    atom_mass = new double[n_atoms];
    
    smart_fgets(line,&line_num,in,source_name,"reading xyz comment");
    for(int i=0;i<n_atoms;i++){
        
        smart_fgets(line,&line_num,in,source_name,"reading atom");
        atom_names[i] = new char[3];
        
        for(start=0;start<strlen(line);start++)if(isalpha(line[start])){
//             fprintf(out_stream,"!%c! %d\n",line[start],start);
            goto read_atom;
        }
        
        fprintf(out_stream,"ERROR: could not find atom name in %s\n",source_name);
        fprintf(out_stream,"       line %d of length %d\n",line_num, strlen(line));
        exit(1);
        
        read_atom:;
        atom_names[i][0]=line[start  ];
        atom_names[i][1]=line[start+1];
        atom_names[i][2]='\0';
        
        sscanf(line+start+2,"%lf %lf %lf %lf",&atom_coord[3*i+0]
                                             ,&atom_coord[3*i+1]
                                             ,&atom_coord[3*i+2]
                                             ,&nucl_charges_full[i]);
        
        nucl_charges_calc[i]=nucl_charges_full[i];
        
        atom_mass   [i]=mass  (atom_names[i], at);
                
        
//         fprintf(out_stream,"%s(%e %e) %e %e %e\n",atom_names[i]
//                                      ,nucl_charges[i]
//                                      ,atom_mass[i]
//                                      ,atom_coord[3*i+0]
//                                      ,atom_coord[3*i+1]
//                                      ,atom_coord[3*i+2]);
//         getchar();
        
    }
    
    for(int i=0;i<3*n_atoms;i++) atom_coord[i]=au_coef*atom_coord[i];
    
    
    
    fclose(in);
    fclose(at);
    delete[] line   ;
    delete[] at_name;
    
    return 0;
}

int molecule::nat_orb_calc(double * gamma, int renorm, char S){
    
//     PrintMatr(gamma,n_act_orb[0],n_act_orb[0],1);
    double f;
    double * NO_VEC_S;
    double * nat_orb_occ_S;
    if(S=='T'){
        f=0.5;
        NO_VEC_S     =NO_VEC     ;
        nat_orb_occ_S=nat_orb_occ;
    }
    else if(S=='A'){
        f=1.0;
        NO_VEC_S     =NO_VEC     ;
        nat_orb_occ_S=nat_orb_occ;
    }
    else if(S=='B'){
        f=1.0;
        NO_VEC_S     =NO_VEC_B     ;
        nat_orb_occ_S=nat_orb_occ_B;
    }
    else{
        fprintf(out_stream,"wrong spin in nat_orb_calc\n");
        exit(0);
    }
    
    memcpy(NO_VEC_S,MO_VEC,sizeof(double)*n_ao*n_cor_orb);
    if(n_act_orb[0]==0) return 0;
    
    if(nat_orbs_m==NULL)nat_orbs_m=new double[n_act_orb[0]*n_act_orb[0]];

    memcpy(nat_orbs_m,gamma,sizeof(double)*n_act_orb[0]*n_act_orb[0]);
//     PrintMatr(nat_orbs_m,n_act_orb[0],n_act_orb[0],1);
    
    lapack_diag(nat_orbs_m, nat_orb_occ_S, n_act_orb[0]);
    for(int i=0;i<n_act_orb[0];i++)if(nat_orb_occ_S[i]<0)nat_orb_occ_S[i]=0;
    if(renorm){
        for(int i=0;i<n_act_orb[0];i++)
        for(int j=0;j<n_act_orb[0];j++)
            nat_orbs_m[i*n_act_orb[0]+j]=nat_orbs_m[i*n_act_orb[0]+j]*sqrt(nat_orb_occ_S[i]*f);
        
//         for(int i=0;i<n_act_orb[0];i++)
//             nat_orb_occ_S[i]=1;
    }
//     PrintMatr(nat_orbs_m,n_act_orb[0],n_act_orb[0],1);
//     fprintf(out_stream,"nat_orb occupations for %c:\n",S);
//     PrintMatr(nat_orb_occ_S,n_act_orb[0],1,1);
    
    nopt_par_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                   n_act_orb[0], n_ao, n_act_orb[0], 1.0,
                   nat_orbs_m, n_act_orb[0],
                   MO_VEC+n_cor_orb*n_ao, n_ao,0.0,
                   NO_VEC_S+n_cor_orb*n_ao,n_ao);
    
//     PrintMatr(NO_VEC_S,n_cor_orb+n_act_orb[0],n_ao,1);
    
    return 0;
}

int molecule::sort_orbs(char t){
    
    int * order = new int[n_mo];
    
    double * BUF_VEC = new double[n_ao*n_mo];
    int * rep_num_buf = new int[n_mo];
    double * energy_buf = new double[n_mo];
    
    find_N_min_els(order, n_mo, orb_energy, n_mo);
    
    if(t=='u'){
        for(int i=0;i<n_mo;i++)
        for(int j=0;j<n_ao;j++)
            BUF_VEC[i*n_ao+j]=MO_VEC[order[i]*n_ao+j];
        
        for(int i=0;i<n_mo;i++)
            rep_num_buf[i]=rep_num[order[i]];
        for(int i=0;i<n_mo;i++)
            energy_buf[i]=orb_energy[order[i]];
    }
    else if(t=='d'){
        for(int i=0;i<n_mo;i++)
        for(int j=0;j<n_ao;j++)
            BUF_VEC[(n_mo-i-1)*n_ao+j]=MO_VEC[order[i]*n_ao+j];
        
        for(int i=0;i<n_mo;i++)
            rep_num_buf[n_mo-i-1]=rep_num[order[i]];
        for(int i=0;i<n_mo;i++)
            energy_buf[n_mo-i-1]=orb_energy[order[i]];
    }
    else{
        fprintf(out_stream,"ERROR: sort_orbs can use only \'u\' or \'d\'\n");
        exit(0);
    }
    
    memcpy(MO_VEC,BUF_VEC,n_mo*n_ao*sizeof(double));
    set_zero_matr(MO_VEC+n_mo*n_ao,(n_ao-n_mo)*n_ao);
    
    for(int i=0;i<n_mo;i++)
        rep_num[i]=rep_num_buf[i];
    for(int i=n_mo;i<n_ao;i++)
        rep_num[i]=-1;
    for(int i=0;i<n_mo;i++)
        orb_energy[i]=energy_buf[i];
    for(int i=n_mo;i<n_ao;i++)
        orb_energy[i]=1000000;
    
    
    
    delete[] BUF_VEC;
    delete[] rep_num_buf;
    delete[] energy_buf;
    delete[] order;
    
    return 0;
}

int molecule::sort_orbs_by_rep(int first, int last){
    
    int n_so=last-first;
    
    int * order = new int[n_so];
    double * rep_num_d = new double[n_ao];
    for(int i=0;i<n_ao;i++)rep_num_d[i]=rep_num[i];
    
    double * BUF_VEC = new double[n_ao*n_so];
    int * rep_num_buf = new int[n_so];
    double * energy_buf = new double[n_so];
    
    find_N_min_els(order, n_so, rep_num_d+first, n_so);
    
    for(int i=0;i<n_so;i++)order[i]+=first;
    
    for(int i=0;i<n_so;i++)
    for(int j=0;j<n_ao;j++)
        BUF_VEC[i*n_ao+j]=MO_VEC[order[i]*n_ao+j];
    
    for(int i=0;i<n_so;i++)
        rep_num_buf[i]=rep_num[order[i]];
    for(int i=0;i<n_so;i++)
        energy_buf[i]=orb_energy[order[i]];
    
//     else if(t=='d'){
//         for(int i=0;i<n_mo;i++)
//         for(int j=0;j<n_ao;j++)
//             BUF_VEC[(n_mo-i-1)*n_ao+j]=MO_VEC[order[i]*n_ao+j];
//         
//         for(int i=0;i<n_mo;i++)
//             rep_num_buf[n_mo-i-1]=rep_num[order[i]];
//         for(int i=0;i<n_mo;i++)
//             energy_buf[n_mo-i-1]=orb_energy[order[i]];
//     }
//     else{
//         fprintf(out_stream,"ERROR: sort_orbs can use only \'u\' or \'d\'\n");
//         exit(0);
//     }
//     
    memcpy(MO_VEC+first*n_ao,BUF_VEC,n_so*n_ao*sizeof(double));
//     set_zero_matr(MO_VEC+n_mo*n_ao,(n_ao-n_mo)*n_ao);
    
    for(int i=0;i<n_so;i++)
        rep_num[i+first]=rep_num_buf[i];
    for(int i=0;i<n_so;i++)
        orb_energy[i+first]=energy_buf[i];
    
    
    delete[] BUF_VEC;
    delete[] rep_num_buf;
    delete[] energy_buf;
    delete[] order;
    
    return 0;
}


int molecule::make_symmetry(){
    
//     if(LINEAR)for(int i_r=0;i_r<S.n_rep;i_r++){
//         
//         S.P[i_r]=new double[n_ao*n_ao];
//         
//         set_zero_matr(S.P[i_r],n_ao*n_ao);
//         
//         for(int i=0;i<n_ao;i++)if(rep_AO_num[i]==i_r)S.P[i_r][i*n_ao+i]=1.0;
//         
//     }
    if(LINEAR)return 0;
    
    int n_at = n_atoms;
    int sy_at=0;
    int sy_err=0;
    double * OR = new double[3];
    double *R = atom_coord;
    
    S.at_refl = new int[n_at*S.n_op];
    //atom reflections
    for(int i_at=0;i_at<n_at;i_at++){
        for(int i_op=0;i_op<S.n_op;i_op++){
            OxR(OR,S.Op_Space[i_op],R+3*i_at);
            sy_at = at_search(OR,R,n_at,1e-8);
            
            S.at_refl[i_at*S.n_op+i_op] = sy_at;
            
            if(sy_at==-1){
                fprintf(out_stream,"ERROR: atom %s{% f, % f, % f} does not have reflection by element %d of group %c%d%c to {% f, % f, % f};\n",
                       atom_names[i_at], R[3*i_at]/au_coef, R[3*i_at+1]/au_coef, R[3*i_at+2]/au_coef,i_op,S.T,S.n,S.t,
                                        OR[3*i_at]/au_coef,OR[3*i_at+1]/au_coef,OR[3*i_at+2]/au_coef);
                sy_err=1;
            }
            else if(nucl_charges_full[i_at]!=nucl_charges_full[sy_at]){
                fprintf(out_stream,"ERROR: atom %s{% f, % f, % f} is reflected to %s{% f, % f, % f} by element %d of group %c%d%c;\n",
                       atom_names[ i_at], R[3* i_at]/au_coef, R[3* i_at+1]/au_coef, R[3* i_at+2]/au_coef,
                       atom_names[sy_at], R[3*sy_at]/au_coef, R[3*sy_at+1]/au_coef, R[3*sy_at+2]/au_coef,
                       i_op,S.T,S.n,S.t);
                sy_err=1;
            }
//             fprintf(out_stream,"atom %d reflected to %d by operation %d\n",i_at,sy_at,i_op);
            
        }//getchar();
        
    }
    delete[] OR;
    if(sy_err)exit(1);
    
    //symmetry operation matrices for S,P,D,F functions
    std::vector<std::vector<double>> Op_shell;
    int l_max=0;
    int n_sh = s.size();
    for(auto &AO: s)
        if(AO.contr[0].l>l_max)l_max=AO.contr[0].l;
        
    Op_shell.resize(l_max+1);
//     fprintf(out_stream,"l = %d\n", l_max);
    
    for(int i_l=0;i_l<l_max+1;i_l++){
        Op_shell[i_l].resize((i_l+1)*(i_l+2)*(i_l+1)*(i_l+2)/4);
    }
//     exit(0);
    //symmetry operation matrices in the AO space
    S.Op_Orb[0]=new double[n_ao*n_ao];
    set_zero_matr(S.Op_Orb[0],n_ao*n_ao);
    for(int i_ao=0;i_ao<n_ao;i_ao++)
        S.Op_Orb[0][i_ao*(n_ao+1)]=1.0;
    
    for(int i_op=1;i_op<S.n_op;i_op++){
        Op_shell[0][0]=1;
        if(l_max>0){
        for(int i=0;i<9;i++)
            Op_shell[1][i]=S.Op_Space[i_op][i];
        
        
        
        }
//         fprintf(out_stream,"space:\n");
//         PrintMatr(S.Op_Space[i_op],3,3,1);
        
        if(l_max>1)
        for(int i=0;i<6;i++)
        for(int j=0;j<6;j++){
            Op_shell[2][i*6+j]=S.Op_Space[i_op][D_L[2*i  ]*3+D_L[2*j  ]]*
                               S.Op_Space[i_op][D_L[2*i+1]*3+D_L[2*j+1]];
                               
//         fprintf(out_stream,"orbital %d %d:\n",D_L[2*j],D_L[2*j+1]);
//         PrintMatr(Op_shell[2].data(),6,6,1);
            if(D_L[2*i]!=D_L[2*i+1])
            Op_shell[2][i*6+j]+=S.Op_Space[i_op][D_L[2*i  ]*3+D_L[2*j+1]]*
                                S.Op_Space[i_op][D_L[2*i+1]*3+D_L[2*j  ]];
            
        }
//         fprintf(out_stream,"orbital:\n");
//         PrintMatr(Op_shell[2].data(),6,6,1);
        
        
        

        if(l_max>2){
            set_zero_matr(Op_shell[3].data(),100);
            for(int j=0;j<10;j++)
            for(int i1=0;i1<3;i1++)
            for(int i2=0;i2<3;i2++)
            for(int i3=0;i3<3;i3++)
            {
                
                int i=F_ind(i1,i2,i3);
//                 fprintf(out_stream,"f(%d,%d,%d)=%d\n",i1,i2,i3,i);
//                 getchar();
                Op_shell[3][i*10+j]+=S.Op_Space[i_op][i1*3+F_L[3*j  ]]*
                                     S.Op_Space[i_op][i2*3+F_L[3*j+1]]*
                                     S.Op_Space[i_op][i3*3+F_L[3*j+2]];
                                    
                
            }
//             fprintf(out_stream,"orbital:\n");
//             PrintMatr(Op_shell[3].data(),10,10,1);
        }

        if(l_max>3){
            set_zero_matr(Op_shell[4].data(),225);
            for(int j=0;j<15;j++)
            for(int i1=0;i1<3;i1++)
            for(int i2=0;i2<3;i2++)
            for(int i3=0;i3<3;i3++)
            for(int i4=0;i4<3;i4++)
            {
                
                int i=G_ind(i1,i2,i3,i4);
                Op_shell[4][i*15+j]+=S.Op_Space[i_op][i1*3+G_L[4*j  ]]*
                                     S.Op_Space[i_op][i2*3+G_L[4*j+1]]*
                                     S.Op_Space[i_op][i3*3+G_L[4*j+2]]*
                                     S.Op_Space[i_op][i4*3+G_L[4*j+3]];
            }
        }
        if(l_max>4){
            set_zero_matr(Op_shell[5].data(),441);
            for(int j=0;j<21;j++)
            for(int i1=0;i1<3;i1++)
            for(int i2=0;i2<3;i2++)
            for(int i3=0;i3<3;i3++)
            for(int i4=0;i4<3;i4++)
            for(int i5=0;i5<3;i5++)
            {
                
                int i=H_ind(i1,i2,i3,i4,i5);
                Op_shell[5][i*21+j]+=S.Op_Space[i_op][i1*3+H_L[5*j  ]]*
                                     S.Op_Space[i_op][i2*3+H_L[5*j+1]]*
                                     S.Op_Space[i_op][i3*3+H_L[5*j+2]]*
                                     S.Op_Space[i_op][i4*3+H_L[5*j+3]]*
                                     S.Op_Space[i_op][i5*3+H_L[5*j+4]];
            }
        }
        if(l_max>5){
            fprintf(out_stream,"ERROR: symmetry is not realized for I shells\n");
            exit(0);
        }
        
        S.Op_Orb[i_op]=new double[n_ao*n_ao];
        set_zero_matr(S.Op_Orb[i_op],n_ao*n_ao);
//         fprintf(out_stream,"%d\n",n_sh);
        int sh_num_for_at=0;
        int prev_at=-1;
        int found_sym_shell;
        int i_ao=0;
        for(int i_sh=0;i_sh<n_sh;i_sh++){
            int n_gto_i=s[i_sh].contr[0].size();
            int i_at = shell_center[i_sh]; 
            if(i_at!=prev_at){
                sh_num_for_at=0;
                prev_at=i_at;
            }
            else
                sh_num_for_at++;
            sy_at = S.at_refl[i_at*S.n_op+i_op];
//             fprintf(out_stream,"a %d - %d (%d)\n",i_at, sy_at,sh_num_for_at);
            int j_sh;
            found_sym_shell=0;
            int sh_num2_for_at=0;
            int n_gto_j;
            int j_ao=0;
            for(j_sh=0;j_sh<n_sh;j_sh++){
                n_gto_j=s[j_sh].contr[0].size();
                if(shell_center[j_sh]==sy_at){
                    if(sh_num2_for_at==sh_num_for_at){
                        found_sym_shell=1;
                        break;
                    }
                    else
                        sh_num2_for_at++;
                }
                j_ao+=n_gto_j;
            }
            if(found_sym_shell==0){
                fprintf(out_stream,"ERROR: shell %d does not have reflection by element %d of group %c%d%c;\n",i_sh,i_op,S.T,S.n,S.t);
                exit(0);
            }
            int n_l = s[i_sh].contr[0].l;// n_gto must be equal
            double * O = S.Op_Orb[i_op]+j_ao*n_ao+i_ao;
//             fprintf(out_stream,"%d (%d) %d(%d)\n",i_ao,n_gto_i,j_ao,n_gto_j);
            for(int i_g=0;i_g<n_gto_i;i_g++)//{
            for(int j_g=0;j_g<n_gto_j;j_g++)//{
                O[i_g*n_ao+j_g]=Op_shell[n_l][i_g*n_gto_j+j_g];
//                 fprintf(out_stream,"% .2e  ",O[i_g*n_ao+j_g]);
//             }
//             fprintf(out_stream,"\n");
//             }
//             getchar();
            i_ao+=n_gto_i;
        }
        
    }
    
    //projectors
    
//     for(int i_r=0;i_r<S.n_rep;i_r++)
    for(int i_r=0;i_r<S.n_rep;i_r++){
        S.P[i_r]=new double[n_ao*n_ao];
        set_zero_matr(S.P[i_r],n_ao*n_ao);
        for(int i   =0;i   <n_ao*n_ao;i++   )
        for(int i_op=0;i_op<S.n_op     ;i_op++)
            S.P[i_r][i]+=S.ch[i_r][i_op]*S.Op_Orb[i_op][i];
        
        double c=S.dim[i_r]/S.n_op;
        for(int i   =0;i   <n_ao*n_ao;i++   )
            S.P[i_r][i]=S.P[i_r][i]*c;
        
        
        
    }
    
    return 0;
}

int molecule::lin_prepare(){
    
    
//     rep_AO_num=new int[n_ao];
    
    int n_wrong_coord=0;
    
    for(int i_a=0; i_a<n_atoms;i_a++){
        if(fabs(atom_coord[3*i_a+0])>1e-8){fprintf(out_stream,"ERROR:atom %d is shifted by X-axis\n",i_a);n_wrong_coord++;}
        if(fabs(atom_coord[3*i_a+1])>1e-8){fprintf(out_stream,"ERROR:atom %d is shifted by Y-axis\n",i_a);n_wrong_coord++;}
    }
    if(n_wrong_coord)exit(0);
    
    double * d_dPhi = new double[n_ao*n_ao];
    
    
    Lambda_AO = new double[n_ao*n_ao];
    set_zero_matr(d_dPhi,n_ao*n_ao);
    
    double L_S[1] =
         {  0.0};
    
    double L_P[9] =
         {  0.0, -1.0,  0.0,
            1.0,  0.0,  0.0,
            0.0,  0.0,  0.0};
        
    double L_D[36] = 
         {  0.0, -2.0,  0.0,  0.0,  0.0, 0.0,
            1.0,  0.0,  0.0, -1.0,  0.0, 0.0,
            0.0,  0.0,  0.0,  0.0, -1.0, 0.0,
            0.0,  2.0,  0.0,  0.0,  0.0, 0.0,
            0.0,  0.0,  1.0,  0.0,  0.0, 0.0,
            0.0 , 0.0,  0.0,  0.0,  0.0, 0.0  };
    
    double L_F[100] = 
         {  0.0, -3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0,
            1.0,  0.0,  0.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0,
            0.0,  0.0,  0.0,  0.0, -2.0,  0.0,  0.0,  0.0,  0.0, 0.0,
            0.0,  2.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0, 0.0,
            0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0, 0.0,
            0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0, 0.0,
            0.0,  0.0,  0.0,  3.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0,
            0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0, 0.0,
            0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0, 0.0,
            0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0 };
           
    double L_G[225] = 
         {   0.0, -4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             1.0,  0.0,  0.0, -3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0, -3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  2.0,  0.0,  0.0,  0.0,  0.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0};
    
    
    double L_H[441] = 
         {   0.0, -5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             1.0,  0.0,  0.0, -4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0, -4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  2.0,  0.0,  0.0,  0.0,  0.0, -3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0, -3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  5.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0};
    int i=0;
    int j=0;
    for(auto &b: s){
        for(auto &c: b.contr){
            
            double * L_X=nullptr;
            if(c.l==0) L_X=L_S;
            if(c.l==1) L_X=L_P;
            if(c.l==2) L_X=L_D;
            if(c.l==3) L_X=L_F;
            if(c.l==4) L_X=L_G;
            if(c.l==5) L_X=L_H;               
            if(c.l>5 ){
                fprintf(out_stream,"LINEAR is not realized for i orbitals\n");
                exit(1);
            }
            int n_sh = c.cartesian_size();
            
            for(int ii=0;ii<n_sh;ii++)
            for(int jj=0;jj<n_sh;jj++)
                d_dPhi[(i+ii)*n_ao+j+jj]=L_X[ii*n_sh+jj];
            
//             if(c.l==0) 
//                 rep_AO_num[i]=0;
//             if(c.l==1){
//                 rep_AO_num[i  ]=1;//X
//                 rep_AO_num[i+1]=2;//Y
//                 rep_AO_num[i+2]=0;//Z
//             }
//             if(c.l==2){
//                 rep_AO_num[i  ]=4;//XX
//                 rep_AO_num[i+1]=3;//XY
//                 rep_AO_num[i+2]=1;//XZ
//                 rep_AO_num[i+3]=4;//YY
//                 rep_AO_num[i+4]=2;//YZ
//                 rep_AO_num[i+5]=0;//YY
//             }
//             if(c.l==3){
//                 rep_AO_num[i  ]=5;//XXX
//                 rep_AO_num[i+1]=6;//XXY
//                 rep_AO_num[i+2]=4;//XXZ
//                 rep_AO_num[i+3]=5;//XYY
//                 rep_AO_num[i+4]=3;//XYZ
//                 rep_AO_num[i+5]=1;//XZZ
//                 rep_AO_num[i+6]=6;//YYY
//                 rep_AO_num[i+7]=4;//YYZ
//                 rep_AO_num[i+8]=2;//YZZ
//                 rep_AO_num[i+9]=0;//ZZZ
//             }
//             if(c.l==4){
//                 rep_AO_num[i   ]=8;//XXXX
//                 rep_AO_num[i+ 1]=7;//XXXY
//                 rep_AO_num[i+ 2]=5;//XXXZ
//                 rep_AO_num[i+ 3]=8;//XXYY
//                 rep_AO_num[i+ 4]=6;//XXYZ
//                 rep_AO_num[i+ 5]=4;//XXZZ
//                 rep_AO_num[i+ 6]=7;//XYYY
//                 rep_AO_num[i+ 7]=5;//XYYZ
//                 rep_AO_num[i+ 8]=3;//XYZZ
//                 rep_AO_num[i+ 9]=1;//XZZZ
//                 rep_AO_num[i+10]=8;//YYYY
//                 rep_AO_num[i+11]=6;//YYYZ
//                 rep_AO_num[i+12]=4;//YYZZ
//                 rep_AO_num[i+13]=2;//YZZZ
//                 rep_AO_num[i+14]=0;//ZZZZ
//             }
//             if(c.l==5){
//                 rep_AO_num[i   ]= 9;//XXXXX
//                 rep_AO_num[i+ 1]=10;//XXXXY
//                 rep_AO_num[i+ 2]= 8;//XXXXZ
//                 rep_AO_num[i+ 3]= 9;//XXXYY
//                 rep_AO_num[i+ 4]= 7;//XXXYZ
//                 rep_AO_num[i+ 5]= 5;//XXXZZ
//                 rep_AO_num[i+ 6]=10;//XXYYY
//                 rep_AO_num[i+ 7]= 8;//XXYYZ
//                 rep_AO_num[i+ 8]= 6;//XXYZZ
//                 rep_AO_num[i+ 9]= 4;//XXZZZ
//                 rep_AO_num[i+10]= 9;//XYYYY
//                 rep_AO_num[i+11]= 7;//XYYYZ
//                 rep_AO_num[i+12]= 5;//XYYZZ
//                 rep_AO_num[i+13]= 3;//XYZZZ
//                 rep_AO_num[i+14]= 1;//XZZZZ
//                 rep_AO_num[i+15]=10;//YYYYY
//                 rep_AO_num[i+16]= 8;//YYYYZ
//                 rep_AO_num[i+17]= 6;//YYYZZ
//                 rep_AO_num[i+18]= 4;//YYZZZ
//                 rep_AO_num[i+19]= 2;//YZZZZ
//                 rep_AO_num[i+20]= 0;//ZZZZZ
//             }
            i+=n_sh;
            j+=n_sh;

        }
    }
    calc_S_AO();
    
//     for(int i=0; i<n_ao;i++)
//     for(int j=0; j<n_ao;j++)
//         d_dPhi[i*n_ao+j]=d_dPhi[i*n_ao+j]*sqrt(S_AO[i*n_ao+i]/S_AO[j*n_ao+j]);
//         
    
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                n_ao,n_ao,n_ao,1.0,
                d_dPhi,n_ao,
                S_AO,n_ao,0.0,
                Lambda_AO,n_ao);
    
    delete[] d_dPhi;
    
    return 0;
}
    

int molecule::check_orb_symmetry(){
    
    if(IS_SYM==0)return 0;
    
    double * PxOrb = new double[n_ao];
    double * SxPxOrb = new double[n_ao];
    double * Orb;
    
    calc_S_AO();
    double a;
//     fprintf(out_stream,"%d %d\n",n_ao,n_mo);
//     getchar();
    if(LINEAR==0)for(int i=0;i<n_mo;i++){
        rep_num[i]=-1;
        Orb=MO_VEC+i*n_ao;
        for(int i_r=0;i_r<S.n_rep;i_r++){
            for(int ii=0;ii<n_ao;ii++)
                PxOrb[ii]=0;
            for(int ii=0;ii<n_ao;ii++)
            for(int jj=0;jj<n_ao;jj++)
                PxOrb[ii]+=S.P[i_r][ii*n_ao+jj]*Orb[jj];
            
            for(int ii=0;ii<n_ao;ii++)
                SxPxOrb[ii]=0;
            for(int ii=0;ii<n_ao;ii++)
            for(int jj=0;jj<n_ao;jj++)
                SxPxOrb[ii]+=S_AO[ii*n_ao+jj]*PxOrb[jj];

            a=cblas_ddot(n_ao, Orb, 1,SxPxOrb, 1);
            
            if(fabs(a-1)<5e-6){
                rep_num[i]=i_r;
                break;
            }
            if(fabs(a)>1e-6){
                fprintf(out_stream,"ERROR: orbital %d has no symmetry\n",i);
//                 getchar();
            }
                
        }
    }
    else for(int i=0;i<n_mo;i++){
        rep_num[i]=-1;
        Orb=MO_VEC+i*n_ao;
        for(int i_r=0;i_r<S.n_rep;i_r++){
            for(int ii=0;ii<n_ao;ii++)
                PxOrb[ii]=0;
            for(int ii=0;ii<n_ao;ii++)
            for(int jj=0;jj<n_ao;jj++)
                PxOrb[ii]+=S_AO[ii*n_ao+jj]*Orb[jj]; //SxO
            
            for(int ii=0;ii<n_ao;ii++)
                SxPxOrb[ii]=0;
            for(int ii=0;ii<n_ao;ii++)
            for(int jj=0;jj<n_ao;jj++)
                SxPxOrb[ii]+=S.T_D5[i_r][jj*n_ao+ii]*PxOrb[jj]; //TxSxO

            a=cblas_ddot(n_ao, SxPxOrb, 1,SxPxOrb, 1);//(TxSxO)^2
//             fprintf(out_stream,"%d %d %e %e \n",i,i_r,a,a-1);
            //T_D5 is not normalized
            if(fabs(a)>1e-6){
                if(rep_num[i]==-1)rep_num[i]=i_r;
                else{
                    rep_num[i]=-1;
                    break;
                }

            }
                
        }
            if(rep_num[i]==-1){fprintf(out_stream,"ERROR: orbital %d has no symmetry\n",i);
            /*getchar();*/}
    }
    
    delete[] PxOrb;
    delete[] SxPxOrb;
    
    return 0;
}

int molecule::MO_backup(){
    
    if(MO_BUF==NULL) MO_BUF=new double[n_ao*n_ao];
    memcpy(MO_BUF,MO_VEC,sizeof(double)*n_ao*n_ao);
    if(IS_SYM){
        if(rep_num_backup==NULL) rep_num_backup=new int[n_ao];
        memcpy(rep_num_backup,rep_num,sizeof(int)*n_ao);
    }
    
    return 0;
}

int molecule::MO_restore(){
    
    if(MO_BUF==NULL) {
        printf("ERROR: no backup orbitals to restore\n");
        exit(0);
    }
    
    memcpy(MO_VEC,MO_BUF,sizeof(double)*n_ao*n_ao);
    
    if(IS_SYM)
        memcpy(rep_num,rep_num_backup,sizeof(int)*n_ao);
    
    return 0;
}


int molecule::sync_shell_to_geom(){
    
    int n_shells=s.size();
    for(int i=0; i<n_shells;i++){
        s[i].O[0]=atom_coord[3*shell_center[i]+0];
        s[i].O[1]=atom_coord[3*shell_center[i]+1];
        s[i].O[2]=atom_coord[3*shell_center[i]+2];
    }
    
    n_shells=ri_s.size();
    for(int i=0; i<n_shells;i++){
        ri_s[i].O[0]=atom_coord[3*ri_shell_center[i]+0];
        ri_s[i].O[1]=atom_coord[3*ri_shell_center[i]+1];
        ri_s[i].O[2]=atom_coord[3*ri_shell_center[i]+2];
    }
    
    return 0;
}


int molecule::resize_CI(int n){
    
    if(n_CI==n){
        printf("WARNING: n_CI(old)=n_CI(new)\n");
        printf("         resize is not needed\n");
        return 0;
    }
    
    if(n_CI!=1){
        printf("ERROR: can not resize CI with n_CI!=1\n");
        printf("       n_CI = %d\n", n_CI);
        exit(0);
    }
    
    n_CI = n;
    
    aldet_data * tmp_CI = new aldet_data[n];
    
    CI[0].print_states(0,n,1);
    
    for(int i=0; i<n; i++){
//         fprintf(stderr,"%d %d %d %d %d %d\n", i, n_act_orb[0],
//                           n_act_el_alp[0], 
//                           n_act_el_bet[0], 1, 
//                           CI[0].mult, 
//                           CI[0].print_number
//                           );
        tmp_CI[i].get_dim(n_act_orb[0],
                          n_act_el_alp[0], 
                          n_act_el_bet[0], 1, 
                          CI[0].mult, 
                          CI[0].print_number
                          );//two sets are needed for CASSCF
//         fprintf(stderr,"s=%d\n",p);
        tmp_CI[i].init_zero_vec(1,0);
//         fprintf(stderr,"A\n");
//         printf("%e\n",CI[0].E_states[0][i]);
        for(int j=0;j<CI[0].Nd;j++)tmp_CI[i].coef[0][j]=CI[0].coef[0][j*CI[0].n_states[0]+i];
//         fprintf(stderr,"b\n");
//         printf("%e\n",CI[0].E_states[0][i]);
        tmp_CI[i].E_states[0][0]=CI[0].E_states[0][i];
//         fprintf(stderr,"c\n");
//         tmp_CI[i].n_states[0]=CI[0].n_states[0];
//         tmp_CI[i].copy_coef(0,CI,CI[0].n_states[0],0,0);
        tmp_CI[i].print_states(0,1,1);

    }
    
    delete[] CI;
    
    CI=tmp_CI;
    
    return 0;
    
}

int molecule::alloc_basic(){
    
    if(S_MO ==nullptr)S_MO = new double[n_ao*n_ao];
    if(S_M05==nullptr)S_M05= new double[n_ao*n_ao];
    if(S_AO ==nullptr)S_AO = new double[n_ao*n_ao];
    if(H_AO ==nullptr)H_AO = new double[n_ao*n_ao];
    if(Dx_AO==nullptr)Dx_AO= new double[n_ao*n_ao];
    if(Dy_AO==nullptr)Dy_AO= new double[n_ao*n_ao];
    if(Dz_AO==nullptr)Dz_AO= new double[n_ao*n_ao];
    if(BUF  ==nullptr)BUF  = new double[n_ao*n_ao*3];
    
    return 1;
}

int molecule::alloc_Q_AO(){
    
    if(Qxx_AO == NULL) Qxx_AO = new double[n_ao*n_ao];
    if(Qxy_AO == NULL) Qxy_AO = new double[n_ao*n_ao];
    if(Qxz_AO == NULL) Qxz_AO = new double[n_ao*n_ao];
    if(Qyy_AO == NULL) Qyy_AO = new double[n_ao*n_ao];
    if(Qyz_AO == NULL) Qyz_AO = new double[n_ao*n_ao];
    if(Qzz_AO == NULL) Qzz_AO = new double[n_ao*n_ao];
    
    return 1;    
}
int molecule::clear_Q_AO(){
    
    delete[] Qxx_AO; Qxx_AO = NULL;
    delete[] Qxy_AO; Qxy_AO = NULL;
    delete[] Qxz_AO; Qxz_AO = NULL;
    delete[] Qyy_AO; Qyy_AO = NULL;
    delete[] Qyz_AO; Qyz_AO = NULL;
    delete[] Qzz_AO; Qzz_AO = NULL;
    
    return 1;    
}

int molecule::gen_1el_data(){
    
    V_nuc=0.0;
    V_nuc=E_nuc_calc(this);

    Dx_nuc=D_nuc_calc(this,0);
    Dy_nuc=D_nuc_calc(this,1);
    Dz_nuc=D_nuc_calc(this,2);
    
    V_nuc-=Dx_nuc*F_x+Dy_nuc*F_y+Dz_nuc*F_z;  ////

    calc_d5_n_ao();
    
    if(PP.size()==0)
        have_ecp = 0;
    else
        have_ecp = 1;
    
    alloc_basic();
    
    if(IS_SYM==1){
        make_symmetry();
    }
    
    if(D5){
        n_mo=n_ao_d5;
        make_D5_matr();
    }
    // else{
    //     n_mo=n_mo;
    // }
    
    calc_S_AO();
    calc_H_AO();
    calc_SM05();
    
    calc_D_AO();
    
    if(print_dispersion+print_quadrupole){
        alloc_Q_AO();
        calc_Q_AO();
    }
    
    
    
    return 1;
}


int molecule::P_calc(){
    
    fprintf(out_stream,"ERROR: P_calc is not transfered from doCI\n");
    exit(1);
//   P_AO = new double[n_ao*n_ao*3];
//   
//   // calculating every S element for GTOs
//   
//   int * n_gto_in_ao  = basis->n_gto_in_ao;
//   int * lxyz1        = basis->lxyz       ;
//   int * ao_center1   = basis->ao_center  ;
//   double *exp_coef1  = basis->exp_coef   ;
//   double *contr_coef1= basis->contr_coef ;
//   double *atom_coord1=atom_coord;
//   
//   
//   
//   for(int i=0;i<n_ao;i++) for(int j=0;j<n_ao;j++)
//   {
//     for(int k=0;k<3;k++) P_AO[(i*n_ao+j)*3+k]=0.0;
//     for(int ii=0;ii<n_gto_in_ao[i];ii++) for(int jj=0;jj<n_gto_in_ao[j];jj++)
//     {
//         
//         
//         for(int l=0;l<3;l++)
//         {
//             int l1[3];
//             l1[0]=lxyz1[i*3+0];
//             l1[1]=lxyz1[i*3+1];
//             l1[2]=lxyz1[i*3+2];
//             
//             int l2[3];
//             l2[0]=lxyz1[j*3+0];
//             l2[1]=lxyz1[j*3+1];
//             l2[2]=lxyz1[j*3+2];
//             l2[l]++;
//             
//             P_AO[(i*n_ao+j)*3+l]-=2.0*exp_coef1[50*j+jj]*
//             S_alp(l1[0],l2[0],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+0],atom_coord1[3*ao_center1[j]+0])*
//             S_alp(l1[1],l2[1],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+1],atom_coord1[3*ao_center1[j]+1])*
//             S_alp(l1[2],l2[2],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+2],atom_coord1[3*ao_center1[j]+2])*
//             contr_coef1[50*i+ii]*contr_coef1[50*j+jj];
//             
//             l2[l]-=2;
//             if(l2[l]>=0)
//             {
//                 P_AO[(i*n_ao+j)*3+l]+=1.0*(l2[l]+1)*
//                 S_alp(l1[0],l2[0],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+0],atom_coord1[3*ao_center1[j]+0])*
//                 S_alp(l1[1],l2[1],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+1],atom_coord1[3*ao_center1[j]+1])*
//                 S_alp(l1[2],l2[2],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+2],atom_coord1[3*ao_center1[j]+2])*
//                 contr_coef1[50*i+ii]*contr_coef1[50*j+jj];
//             }
//         }
//     }
//   }
  
  return 0;

}

int molecule::RxP_calc(){
    
    fprintf(out_stream,"ERROR: RxP_calc is not transfered from doCI\n");
    exit(1);
    
//   RxP_AO = new double[n_ao*n_ao*9];
//   
//   double *SxP1_mat = new double[n_ao*n_ao*3];
//   
//   int * n_gto_in_ao  = basis->n_gto_in_ao;
//   int * lxyz1        = basis->lxyz       ;
//   int * ao_center1   = basis->ao_center  ;
//   double *exp_coef1  = basis->exp_coef   ;
//   double *contr_coef1= basis->contr_coef ;
//   double *atom_coord1=atom_coord;
//   
//   
//   
//   // calculating every S element for GTOs
//   
//   for(int i=0;i<n_ao;i++) for(int j=0;j<n_ao;j++)
//   {
//     for(int k=0;k<3;k++) SxP1_mat[(i*n_ao+j)*3+k]=0.0;
//     for(int k=0;k<9;k++) RxP_AO[(i*n_ao+j)*9+k]=0.0;
//     for(int ii=0;ii<n_gto_in_ao[i];ii++) for(int jj=0;jj<basis->n_gto_in_ao[j];jj++)
//     {
//         // s p
//         
//         for(int l=0;l<3;l++)
//         {
//             int l1[3];
//             l1[0]=lxyz1[i*3+0];
//             l1[1]=lxyz1[i*3+1];
//             l1[2]=lxyz1[i*3+2];
//             
//             int l2[3];
//             l2[0]=lxyz1[j*3+0];
//             l2[1]=lxyz1[j*3+1];
//             l2[2]=lxyz1[j*3+2];
//             l2[l]++;
//             
//             SxP1_mat[(i*n_ao+j)*3+l]-=2.0*exp_coef1[50*j+jj]*
//             S_alp(l1[0],l2[0],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+0],atom_coord1[3*ao_center1[j]+0])*
//             S_alp(l1[1],l2[1],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+1],atom_coord1[3*ao_center1[j]+1])*
//             S_alp(l1[2],l2[2],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+2],atom_coord1[3*ao_center1[j]+2])*
//             contr_coef1[50*i+ii]*contr_coef1[50*j+jj];
//             
//             l2[l]-=2;
//             if(l2[l]>=0)
//             {
//                 SxP1_mat[(i*n_ao+j)*3+l]+=1.0*(l2[l]+1)*
//                 S_alp(l1[0],l2[0],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+0],atom_coord1[3*ao_center1[j]+0])*
//                 S_alp(l1[1],l2[1],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+1],atom_coord1[3*ao_center1[j]+1])*
//                 S_alp(l1[2],l2[2],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+2],atom_coord1[3*ao_center1[j]+2])*
//                 contr_coef1[50*i+ii]*contr_coef1[50*j+jj];
//             }
//         }
//         
//         // r p
//         
//         for(int k=0;k<3;k++) for(int l=0;l<3;l++)
//         {
//             int l1[3];
//             l1[0]=lxyz1[i*3+0];
//             l1[1]=lxyz1[i*3+1];
//             l1[2]=lxyz1[i*3+2];
//             
//             l1[k]++;
//             
//             int l2[3];
//             l2[0]=lxyz1[j*3+0];
//             l2[1]=lxyz1[j*3+1];
//             l2[2]=lxyz1[j*3+2];
//             l2[l]++;
//             
//             RxP_AO[(i*n_ao+j)*9+3*k+l]-=2.0*exp_coef1[50*j+jj]*
//             S_alp(l1[0],l2[0],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+0],atom_coord1[3*ao_center1[j]+0])*
//             S_alp(l1[1],l2[1],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+1],atom_coord1[3*ao_center1[j]+1])*
//             S_alp(l1[2],l2[2],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+2],atom_coord1[3*ao_center1[j]+2])*
//             contr_coef1[50*i+ii]*contr_coef1[50*j+jj];
//             
//             l2[l]-=2;
//             
//             if(l2[l]>=0)
//             {
//                 RxP_AO[(i*n_ao+j)*9+3*k+l]+=1.0*(l2[l]+1)*
//                 S_alp(l1[0],l2[0],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+0],atom_coord1[3*ao_center1[j]+0])*
//                 S_alp(l1[1],l2[1],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+1],atom_coord1[3*ao_center1[j]+1])*
//                 S_alp(l1[2],l2[2],exp_coef1[50*i+ii],exp_coef1[50*j+jj],atom_coord1[3*ao_center1[i]+2],atom_coord1[3*ao_center1[j]+2])*
//                 contr_coef1[50*i+ii]*contr_coef1[50*j+jj];
//             }
//         }
//     }
//   }
//   
//   for(int i=0;i<n_ao;i++) for(int j=0;j<n_ao;j++) for(int k=0;k<3;k++) for(int l=0;l<3;l++) 
//       RxP_AO[(i*n_ao+j)*9+k*3+l]+=SxP1_mat[(i*n_ao+j)*3+l]*atom_coord1[3*ao_center1[i]+k];
//   
//   delete[] SxP1_mat;
//   
  return 0;

}

int molecule::diag_X_AO_in_MO(double * X){
    
    if(LINEAR){
        if(D5==0){
            printf("ERROR: LINEAR requires D5\n");
            exit(0);
        }
        double* tmp_ev = new double[n_ao];
        double* V       = new double[n_ao*n_ao];
        double* V2      = new double[2*n_ao*n_ao];
        int n_eig;
        set_zero_matr(MO_VEC,n_ao*n_ao);
        int n_0=0;
        for(int i_r=0;i_r<S.n_rep;i_r++){
            
            
            double * P=S.T_D5[i_r];
               
            int n_mo=n_ao;
            if(S.n_ao_d5[i_r]==0)n_eig=0;
            else n_eig = HPC_SPCE (X , S_AO, P, V, V2, tmp_ev, n_ao);
            for(int i=0;i<n_eig;i++){
                orb_energy[i+n_0]=tmp_ev[n_eig-1-i];
                rep_num[i+n_0]=0;
                for(int j=0;j<n_mo;j++) MO_VEC[(n_eig-1-i+n_0)*n_mo+j]=V[i*n_mo+j];                
            }
            n_0+=n_eig;
        }
        sort_orbs('u');
//         MO_gamess_format();
//         GAMESS_type_out_print("tmp.out",-1);
//         exit(0);
        check_orb_symmetry();
        
        delete[] tmp_ev;
        delete[] V     ;
        delete[] V2    ;
        
        
    }
    else if(IS_SYM==0){
        if(D5==0){
            tr_and_diag(X, S_M05, BUF, MO_VEC, orb_energy, n_ao);
        }
        else{
            double * S_D5 = BUF+n_ao*n_ao  ;
            double * V    = BUF+n_ao*n_ao*2;
            
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        n_ao,n_ao_d5,n_ao,1.0,
                        X   ,n_ao,
                        T_D5,n_ao_d5,0.0,
                        BUF ,n_ao_d5);
            cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                        n_ao_d5,n_ao_d5,n_ao,1.0,
                        T_D5,n_ao_d5,
                        BUF ,n_ao_d5,0.0,
                        X  ,n_ao_d5);
            
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        n_ao,n_ao_d5,n_ao,1.0,
                        S_AO,n_ao,
                        T_D5,n_ao_d5,0.0,
                        BUF ,n_ao_d5);
            cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                        n_ao_d5,n_ao_d5,n_ao,1.0,
                        T_D5,n_ao_d5,
                        BUF ,n_ao_d5,0.0,
                        S_D5,n_ao_d5);
//             
//             printf("H:\n");
//             PrintMatr(X,n_ao_d5,n_ao_d5,1);
//             printf("S:\n");
//             PrintMatr(S_D5,n_ao_d5,n_ao_d5,1);
            
            
            HC_SCE(X, S_D5, V, BUF, orb_energy, n_ao_d5);
                        
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        n_ao_d5,n_ao,n_ao_d5,1.0,
                        V      ,n_ao_d5,
                        T_D5,n_ao_d5,0.0,
                        MO_VEC,n_ao);
        
            
//             exit(0);
        }
    
    }
    else{
        double* tmp_ev = new double[n_ao];
        double* V       = new double[n_ao*n_ao];
        double* V2      = new double[2*n_ao*n_ao];
        
        int n_0=0;
        double * P=nullptr;
        if(D5){
            P=new double[n_ao*n_ao];
            set_zero_matr(P,n_ao*n_ao);
        }
        for(int i_r=0;i_r<S.n_rep;i_r++){
            int n_mo=n_ao;
            if(D5==0)P=S.P[i_r];
            else{
                cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        n_ao,n_ao_d5,n_ao,1.0,
                        S.P[i_r] ,n_ao,
                        T_D5,n_ao_d5,0.0,
                        P  ,n_ao);
            
            }    
            
            int n_eig = HPC_SPCE (X , S_AO, P, V, V2, tmp_ev, n_ao);
            for(int i=0;i<n_eig;i++){
                orb_energy[i+n_0]=tmp_ev[n_eig-1-i];
                rep_num[i+n_0]=i_r;
                for(int j=0;j<n_mo;j++) MO_VEC[(n_eig-1-i+n_0)*n_mo+j]=V[i*n_mo+j];                
            }
            n_0+=n_eig;
        }
        if(D5){
            delete[] P;
        }

        sort_orbs('u');
        check_orb_symmetry();
        
        delete[] tmp_ev;
        delete[] V     ;
        delete[] V2    ;
        
    }
    
    return 0;
}

int molecule::diag_X_MO_block(double * X, int n0, int dim, double * U){
    
    if(dim==0)return 0;
    
    // Under $SYMM a block holding a row without an irrep label is diagonalized whole and
    // relabelled: the per-irrep branch would drop that row.
    bool all_labelled = true;
    if(IS_SYM)for(int i=0;i<dim;i++)if(rep_num[i+n0]<0||rep_num[i+n0]>=S.n_rep)all_labelled=false;
    
    if(IS_SYM==0||!all_labelled){
        double * F;
        if(U==nullptr)F = new double[dim*dim];
        else          F = U;
        double * B  = new double[dim*n_ao];
        double * ev = orb_energy+n0;
        for(int i=0;i<dim;i++)
        for(int j=0;j<dim;j++)
            F[i*dim+j]=X[(i+n0)*n_ao+j+n0];
        
        lapack_diag(F,ev,dim);
        normalize_rotation_rows(F, dim);
        
//         PrintMatr(ev,dim,1,0);
//         exit(0);
        
        for(int i=0;i<dim;i++)
        for(int j=0;j<dim;j++)
            X[(i+n0)*n_ao+j+n0]=0;
        
        for(int i=0;i<dim;i++)
            X[(i+n0)*n_ao+i+n0]=ev[i];
        
        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                         dim,n_ao,dim,1.0,
                         F,dim,
                         MO_VEC+n0*n_ao,n_ao,0.0,
                         B,n_ao);
        memcpy(MO_VEC+n0*n_ao,B,dim*n_ao*sizeof(double));
        
        if(U==nullptr)delete[] F;
        // delete[] ev;
        delete[] B;
        if(IS_SYM)check_orb_symmetry();
    }
    else{
        int n_rep=S.n_rep;
        int * dim_r    = new int   [n_rep];
        double ** F    = new double*[n_rep];
        double ** B    = new double*[n_rep];
        // double ** ev_r = new double*[n_rep];
        
        double *  ev = orb_energy+n0;
        
//         printf("dim=%d\n",dim);
        double * U_r = U;
        for(int i_r=0;i_r<n_rep;i_r++){
            dim_r[i_r]=0;
            for(int i=0;i<dim;i++)if(rep_num[i+n0]==i_r)dim_r[i_r]++;
            F [i_r] =new double[dim_r[i_r]*dim_r[i_r]];
            B [i_r] =new double[dim_r[i_r]*n_ao];
            
            if(dim_r[i_r]!=0){
                int ii=0;
                for(int i=0;i<dim;i++)if(rep_num[i+n0]==i_r){
                    
                    for(int j=0;j<n_ao;j++)B[i_r][ii*n_ao+j]=MO_VEC[(i+n0)*n_ao+j];
                    
                    int jj=0;
                    for(int j=0;j<dim;j++)if(rep_num[j+n0]==i_r){
                        F[i_r][ii*dim_r[i_r]+jj]=X[(i+n0)*n_ao+j+n0];
                        jj++;
                    }
                    ii++;
                }
                lapack_diag(F[i_r],ev,dim_r[i_r]);
                normalize_rotation_rows(F[i_r], dim_r[i_r]);
                if(U!=nullptr){
                    
                    for(int i=0;i<dim_r[i_r];i++){
                        int jj=0;
                        for(int j=0;j<dim;j++)if(rep_num[j+n0]==i_r){
                            U_r[i*dim+j]=F[i_r][i*dim_r[i_r]+jj];
                            jj++;
                        }
                        else U_r[i*dim+j]=0;
                    
                    }
                    U_r+=dim_r[i_r]*dim;
                }
                
            }
            ev+=dim_r[i_r];
        }
        for(int i=0;i<dim;i++)
        for(int j=0;j<dim;j++)
            X[(i+n0)*n_ao+j+n0]=0;
        for(int i=0;i<dim;i++)
            X[(i+n0)*n_ao+i+n0]=orb_energy[i+n0];
        
        double *MO_c_VEC = MO_VEC+n0*n_ao;
        int j0=n0;
        
        for(int i_r=0;i_r<n_rep;i_r++){
            
            for(int j=0; j<dim_r[i_r];j++)rep_num[j+j0]=i_r;
            j0+=dim_r[i_r];
            
            if(dim_r[i_r]!=0)
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                             dim_r[i_r],n_ao,dim_r[i_r],1.0,
                             F[i_r],dim_r[i_r],
                             B[i_r],n_ao,0.0,
                             MO_c_VEC,n_ao);
            MO_c_VEC+=dim_r[i_r]*n_ao;
            
            delete[] F [i_r];
            delete[] B [i_r];
        }
        
        delete[] dim_r;
        delete[] F    ;
        delete[] B    ;
        // delete[] ev_r ;
        // delete[] ev   ;
        
    }
    
    if(U!=nullptr)transpose(U,dim,dim);//transpose for further malmquist transform of CI vector
    
    return 0;    
    
}

int molecule::calc_S_AO(){
    
    if(S_AO ==nullptr)S_AO = new double[n_ao*n_ao];
    set_zero_matr(S_AO,n_ao*n_ao);
    AO_1el_from_2shells(S_AO,s,s,n_ao,n_ao,'s',0);
        
    return 0;
    
}

int molecule::calc_SM05(){
    
    memcpy(BUF, S_AO, n_ao*n_ao*sizeof(double));
    double * SP05 = BUF+n_ao*n_ao;
    
    S05_calc(BUF,S_M05,SP05,n_ao);
    
    return 0;
}

int molecule::MO_orth(){
    double * SP05 = new double[n_ao*n_ao];
    double * BUF2 = BUF+n_ao*n_ao;
    double * BUF3 = BUF+2*n_ao*n_ao;
    double * ev = new double[n_ao];
    
    memcpy(BUF, S_AO, n_ao*n_ao*sizeof(double));
    
    S05_calc(BUF,S_M05,SP05,n_ao);
    
    int n_occ_o=n_cor_orb;
    for(int i=0;i<n_frag;i++){
        n_occ_o+=n_act_orb[i];
    }
    
    int n_d6;
    if(D5){
        
        make_D5_matr();
        n_d6=n_ao-n_ao_d5;
        
        
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,
                              n_ao_d5,n_ao,n_ao,1.0,
                              T_D5,n_ao_d5,
                              SP05,n_ao,0.0,
                              BUF,n_ao);
        
        ortogonalization_no_sq(BUF,n_ao_d5,n_ao);
        
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                              n_ao,n_ao,n_ao_d5,1.0,
                              BUF,n_ao,
                              BUF,n_ao,0.0,
                              BUF3,n_ao);
        
        
        
        lapack_diag(BUF3,ev,n_ao);
        
        memcpy(V_D6,BUF3,n_d6*n_ao*sizeof(double));
    }
    else
        n_d6=0;
    
    double * SP05_MO = new double[(n_occ_o+n_d6)*n_ao];
    set_zero_matr(SP05_MO,(n_occ_o+n_d6)*n_ao);
    
    // S^(1/2)*D6-orbitals -> V
    memcpy(SP05_MO, V_D6, n_d6*n_ao*sizeof(double));
    
    // S^(1/2)*(core+active) -> V
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                          n_occ_o,n_ao,n_ao,1.0,
                          MO_VEC,n_ao,
                          SP05,n_ao,0.0,
                          SP05_MO+n_d6*n_ao,n_ao);
    
    // S^(1/2)*(D6+core+active) -> orthogonalization
    ortogonalization_no_sq(SP05_MO,n_occ_o+n_d6,n_ao);
    
    // projector on D6+core+active
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                          n_ao,n_ao,n_occ_o+n_d6,1.0,
                          SP05_MO,n_ao,
                          SP05_MO,n_ao,0.0,
                          BUF3,n_ao);
    
    
    double * CA_proj = new double[n_ao*n_ao];
    // D'=S^(1/2) D S^(1/2)
    transform(CA_proj,BUF3,SP05,BUF2,n_ao);
    n_mo=n_ao-n_d6;
    set_zero_matr(MO_VEC,n_ao*n_ao);
    // diagonalization with ir.rep classification
    diag_X_AO_in_MO(CA_proj);
    sort_orbs('d');
    
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                          n_occ_o,n_ao,n_ao,1.0,
                          SP05_MO+n_d6*n_ao,n_ao,
                          S_M05,n_ao,0.0,
                          MO_VEC,n_ao);
    
    // MO_gamess_format();
    // GAMESS_type_out_print("X.out",-1);
    
    
    delete[] CA_proj;
    delete[] SP05;
    delete[] SP05_MO;
    
    check_orb_symmetry();
        
    return 0;
}


int molecule::calc_H_AO(){
    
    if(backup.h1==0){
        read_H_AO();
    }    
    
    if(backup.h1){
        set_zero_matr(H_AO,n_ao*n_ao);
        AO_1el_from_2shells(H_AO,s,s,n_ao,n_ao,'t',0);
        AO_1el_from_2shells(H_AO,s,s,n_ao,n_ao,'v',0);
        if(have_ecp){
#ifdef _USE_GRPP
            grpp_engine.calc_H_AO(H_AO);
#endif
#ifndef _USE_GRPP
            printf("ERROR: ECP calculation is not supported\n");
            printf("       recompile NOPT with LIBGRPP\n");
            exit(1);
#endif
        }
    }
        
    
    if(backup.h1==2){
        write_H_AO();
        backup.h1=0;
    }
    
    return 0;
}


int molecule::write_H_AO(){
        
    binary_array_write(H_AO,"H_1el.dat", backup.prefix,n_ao*n_ao, 0);
    
    return 0;
    
}

int molecule::read_H_AO(){
    
    int read_failure;
    read_failure = binary_array_read(H_AO,"H_1el.dat", backup.prefix,n_ao*n_ao);
    if(read_failure)backup.h1=2;
    
    return 0;
}

int molecule::calc_F_AO(double * F, double * DM, double c){
    
    double *J[1];
    double *K[1];
    J[0] = new double[n_ao*n_ao];
    K[0] = new double[n_ao*n_ao];
    
    set_zero_matr(J[0],n_ao*n_ao);
    set_zero_matr(K[0],n_ao*n_ao);
    
    if(RI==0) {
        DM_to_F_transform(J, K, &DM, 1, &DM, 1, s, n_ao);
        for(int i=0;i<n_ao*n_ao;i++)F[i] = H_AO[i]+(2*J[0][i]-K[0][i])*c;
    }
    
    if(RI!=0){
        if(mc==0)DM_to_F_transform_RI(J[0], K[0], DM, MO_VEC, n_el_calc/2              , n_ao);
        if(mc==1)DM_to_F_transform_RI(J[0], K[0], DM, NO_VEC, n_cor_orb+n_act_orb[0], n_ao);
        if(mc==2)DM_to_F_transform_RI(J[0], K[0], DM, NO_VEC, n_cor_orb                , n_ao);
        
        if(mc==0)for(int i=0;i<n_ao*n_ao;i++)F[i] = H_AO[i]+(2*J[0][i]-K[0][i])*c;
        if(mc==2)for(int i=0;i<n_ao*n_ao;i++)F[i] = H_AO[i]+(2*J[0][i]-K[0][i])*c;
        c = 2.0*c;
        if(mc==1)for(int i=0;i<n_ao*n_ao;i++)F[i] = H_AO[i]+(  J[0][i]-K[0][i])*c;// J is generated by DM and K by NO_VEC
    }
    
    delete[] J[0];
    delete[] K[0];

    
    return 1;
}

int molecule::calc_U_F_AO(double * F_A,double * F_B, double * DM_A, double * DM_B, double c){
    
    double *J[2];
    double *K[2];
    J[0] = new double[n_ao*n_ao];
    K[0] = new double[n_ao*n_ao];
    J[1] = new double[n_ao*n_ao];
    K[1] = new double[n_ao*n_ao];
    
    set_zero_matr(J[0],n_ao*n_ao);
    set_zero_matr(K[0],n_ao*n_ao);
    
    if(RI==0) {
        fprintf(out_stream,"U_F_AO is not realized for RI=0\n");
        exit(0);
    }
    
    if(RI!=0){
        if(mc==0){fprintf(out_stream,"U_F_AO is not realized for mc=0\n");exit(0);}
        if(mc==2){fprintf(out_stream,"U_F_AO is not realized for mc=2\n");exit(0);}
        
        
//         if(mc==0)DM_to_F_transform_RI(J[0], K[0], DM, MO_VEC, n_el/2                   , n_ao);
        if(mc==1)DM_to_F_transform_RI(J[0], K[0], DM_A, NO_VEC  , n_cor_orb+n_act_orb[0], n_ao);
        if(mc==1)DM_to_F_transform_RI(J[1], K[1], DM_B, NO_VEC_B, n_cor_orb+n_act_orb[0], n_ao);
//         if(mc==2)DM_to_F_transform_RI(J[0], K[0], DM_A, NO_VEC, n_cor_orb                , n_ao);
//         if(mc==2)DM_to_F_transform_RI(J[1], K[1], DM_B, NO_VEC, n_cor_orb                , n_ao);
//         if(mc==0)for(int i=0;i<n_ao*n_ao;i++)F[i] = H_AO[i]+(2*J[0][i]-K[0][i])*c;
//         if(mc==2)for(int i=0;i<n_ao*n_ao;i++)F_A[i] = H_AO[i]+(J[0][i]+J[1][i]-K[0][i])*c;
//         if(mc==2)for(int i=0;i<n_ao*n_ao;i++)F_B[i] = H_AO[i]+(J[0][i]+J[1][i]-K[1][i])*c;
//         c = 2.0*c;
//         if(mc==1)for(int i=0;i<n_ao*n_ao;i++)F[i] = H_AO[i]+(  J[0][i]-K[0][i])*c;
        for(int i=0;i<n_ao*n_ao;i++)F_A[i] = H_AO[i]+(J[0][i]+J[1][i]-K[0][i])*c;
        for(int i=0;i<n_ao*n_ao;i++)F_B[i] = H_AO[i]+(J[0][i]+J[1][i]-K[1][i])*c;
    }
    
    
//     fprintf(out_stream,"F\n:");
//     PrintMatr(F, n_ao, n_ao,0);
//     exit(0);
    
    delete[] J[0];
    delete[] K[0];
    delete[] J[1];
    delete[] K[1];

    
    return 1;
}


int molecule::calc_D_AO(){
    
    if(print_dipole==0)
        return 1;
    
    
    set_zero_matr(Dx_AO,n_ao*n_ao);
    set_zero_matr(Dy_AO,n_ao*n_ao);
    set_zero_matr(Dz_AO,n_ao*n_ao);
    AO_1el_from_2shells(Dx_AO,s,s,n_ao,n_ao,'d',1);
    AO_1el_from_2shells(Dy_AO,s,s,n_ao,n_ao,'d',2);
    AO_1el_from_2shells(Dz_AO,s,s,n_ao,n_ao,'d',3);
    for(int i=0; i<n_ao*n_ao;i++)Dx_AO[i]=-Dx_AO[i];
    for(int i=0; i<n_ao*n_ao;i++)Dy_AO[i]=-Dy_AO[i];
    for(int i=0; i<n_ao*n_ao;i++)Dz_AO[i]=-Dz_AO[i];
    for(int ii=0;ii<n_ao*n_ao;ii++)
        H_AO[ii]-= Dx_AO[ii]*F_x+Dy_AO[ii]*F_y+Dz_AO[ii]*F_z; ////

    return 1;
}

int molecule::calc_Q_AO(){
    
    if(print_quadrupole==0)
    if(print_dispersion==0)
        return 1;
    
    set_zero_matr(Qxx_AO,n_ao*n_ao);
    set_zero_matr(Qxy_AO,n_ao*n_ao);
    set_zero_matr(Qxz_AO,n_ao*n_ao);
    set_zero_matr(Qyy_AO,n_ao*n_ao);
    set_zero_matr(Qyz_AO,n_ao*n_ao);
    set_zero_matr(Qzz_AO,n_ao*n_ao);
    
    AO_1el_from_2shells(Qxx_AO,s,s,n_ao,n_ao,'q',4);
    AO_1el_from_2shells(Qxy_AO,s,s,n_ao,n_ao,'q',5);
    AO_1el_from_2shells(Qxz_AO,s,s,n_ao,n_ao,'q',6);
    AO_1el_from_2shells(Qyy_AO,s,s,n_ao,n_ao,'q',7);
    AO_1el_from_2shells(Qyz_AO,s,s,n_ao,n_ao,'q',8);
    AO_1el_from_2shells(Qzz_AO,s,s,n_ao,n_ao,'q',9);
    
    
    return 1;
}

int molecule::calc_d5_n_ao(){
    
    n_ao_d5=0;
    for(auto &b: s){
        for(auto &c: b.contr){
            n_ao_d5+=2*c.l+1;
//             fprintf(out_stream,"%d %d %d\n",n_ao_d5,2*c.l+1,c.l);
//             getchar();
        }
    }
    
    return 0;

}

int molecule::make_D5_matr(){
    
    double D5M[30] = 
         {  0.0,  1.0,  0.0,  0.0,  0.0, 0.0,
            0.0,  0.0,  0.0,  0.0,  1.0, 0.0,
           -1.0,  0.0,  0.0, -1.0,  0.0, 2.0,
            0.0,  0.0,  1.0,  0.0,  0.0, 0.0,
            1.0 , 0.0,  0.0, -1.0,  0.0, 0.0  };
    
    double F7M[70] = 
         {  0.0,  3.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0, 0.0,
            0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0, 0.0,
            0.0, -1.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  4.0, 0.0,
            0.0,  0.0, -1.5,  0.0,  0.0,  0.0,  0.0, -1.5,  0.0, 1.0,
           -1.0 , 0.0,  0.0, -1.0,  0.0,  4.0,  0.0,  0.0,  0.0, 0.0,
            0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0, 0.0,
            1.0 , 0.0,  0.0, -3.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0 };
           
    double G9M[135] = 
         {   0.0,  1.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,     0.0,  0.0,  0.0,     0.0, 0.0,
             0.0,  0.0,  0.0,  0.0,  3.0,  0.0,  0.0,  0.0,  0.0,  0.0,     0.0, -1.0,  0.0,     0.0, 0.0,
             0.0, -1.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  6.0,  0.0,     0.0,  0.0,  0.0,     0.0, 0.0,
             0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,     0.0, -1.0,  0.0, 4.0/3.0, 0.0,
             1.0,  0.0,  0.0,  2.0,  0.0, -8.0,  0.0,  0.0,  0.0,  0.0,     1.0,  0.0, -8.0,     0.0, 8.0/3.0,
             0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0, -1.0,  0.0, 4.0/3.0,  0.0,  0.0,  0.0,     0.0, 0.0,
            -1.0,  0.0,  0.0,  0.0,  0.0,  6.0,  0.0,  0.0,  0.0,  0.0,     1.0,  0.0, -6.0,     0.0, 0.0,
             0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0, -3.0,  0.0,  0.0,     0.0,  0.0,  0.0,     0.0, 0.0,
             1.0,  0.0,  0.0, -6.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,     1.0,  0.0,  0.0,     0.0, 0.0};
    
    
    double H11M[231] = 
         {  0.0,  5.0,   0.0,   0.0,  0.0,   0.0, -10.0,  0.0,   0.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.0,  1.0,   0.0,   0.0,  0.0,  0.0, 0.0,
            0.0,  0.0,   0.0,   0.0,  1.0,   0.0,   0.0,  0.0,   0.0,  0.0,  0.0, -1.0,   0.0,  0.0,  0.0,  0.0,   0.0,   0.0,  0.0,  0.0, 0.0,
            0.0, -3.0,   0.0,   0.0,  0.0,   0.0,  -2.0,  0.0,  24.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.0,  1.0,   0.0,  -8.0,  0.0,  0.0, 0.0,
            0.0,  0.0,   0.0,   0.0, -1.0,   0.0,   0.0,  0.0,   0.0,  0.0,  0.0, -1.0,   0.0,  2.0,  0.0,  0.0,   0.0,   0.0,  0.0,  0.0, 0.0,
            0.0,  1.0,   0.0,   0.0,  0.0,   0.0,   2.0,  0.0, -12.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.0,  1.0,   0.0, -12.0,  0.0,  8.0, 0.0,
            0.0,  0.0, 1.875,   0.0,  0.0,   0.0,   0.0, 3.75,   0.0, -5.0,  0.0,  0.0,   0.0,  0.0,  0.0,  0.0, 1.875,   0.0, -5.0,  0.0, 1.0,
            1.0,  0.0,   0.0,   2.0,  0.0, -12.0,   0.0,  0.0,   0.0,  0.0,  1.0,  0.0, -12.0,  0.0,  8.0,  0.0,   0.0,   0.0,  0.0,  0.0, 0.0,
            0.0,  0.0,  -1.0,   0.0,  0.0,   0.0,   0.0,  0.0,   0.0,  2.0,  0.0,  0.0,   0.0,  0.0,  0.0,  0.0,   1.0,   0.0, -2.0,  0.0, 0.0,
           -1.0,  0.0,   0.0,   2.0,  0.0,   8.0,   0.0,  0.0,   0.0,  0.0,  3.0,  0.0, -24.0,  0.0,  0.0,  0.0,   0.0,   0.0,  0.0,  0.0, 0.0,
            0.0,  0.0,   1.0,   0.0,  0.0,   0.0,   0.0, -6.0,   0.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.0,  0.0,   1.0,   0.0,  0.0,  0.0, 0.0,
            1.0,  0.0,   0.0, -10.0,  0.0,   0.0,   0.0,  0.0,   0.0,  0.0,  5.0,  0.0,   0.0,  0.0,  0.0,  0.0,   0.0,   0.0,  0.0,  0.0, 0.0 };
    
    double I13M[364] =
         {
            0.0,  1.0 , 0.0,  0.0,  0.0,  0.0,-10.0/3.0,  0.0,  0.0,  0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,     0.0, 0.0, 0.0,
            0.0,  0.0,  0.0,  0.0,  5.0,  0.0,  0.0,      0.0,  0.0,  0.0,   0.0,-10.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  1.0,  0.0,  0.0,     0.0, 0.0, 0.0,
            0.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,      0.0, 10.0,  0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,-10.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,     0.0, 0.0, 0.0,
            0.0,  0.0,  0.0,  0.0, -3.0,  0.0,  0.0,      0.0,  0.0,  0.0,   0.0, -2.0,  0.0,  8.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  1.0,  0.0, -8.0/3.0, 0.0, 0.0, 0.0,
            0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  2.0,      0.0,-16.0,  0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,-16.0,  0.0, 16.0, 0.0,  0.0,  0.0,  0.0,  0.0,     0.0, 0.0, 0.0,
            0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,      0.0,  0.0,  0.0,   0.0,  2.0,  0.0, -4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  1.0,  0.0, -4.0,     0.0, 1.6, 0.0,
           -1.0,  0.0,  0.0, -3.0,  0.0, 18.0,  0.0,      0.0,  0.0,  0.0,  -3.0,  0.0, 36.0,  0.0,-24.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0, -1.0,  0.0, 18.0,  0.0,   -24.0, 0.0, 3.2,
            0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,      2.0,  0.0, -4.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0, -4.0,  0.0, 1.6,  0.0,  0.0,  0.0,  0.0,     0.0, 0.0, 0.0,
            1.0,  0.0,  0.0,  1.0,  0.0,-16.0,  0.0,      0.0,  0.0,  0.0,  -1.0,  0.0,  0.0,  0.0, 16.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0, -1.0,  0.0, 16.0,  0.0,   -16.0, 0.0, 0.0,
            0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0,      2.0,  0.0,8.0/3.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  3.0,  0.0, -8.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,     0.0, 0.0, 0.0,
           -1.0,  0.0,  0.0,  5.0,  0.0, 10.0,  0.0,      0.0,  0.0,  0.0,   5.0,  0.0,-60.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0, -1.0,  0.0, 10.0,  0.0,     0.0, 0.0, 0.0,
            0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,    -10.0,  0.0,  0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  5.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0,     0.0, 0.0, 0.0,
            1.0,  0.0,  0.0,-15.0,  0.0,  0.0,  0.0,      0.0,  0.0,  0.0,  15.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0, -1.0,  0.0,  0.0,  0.0,     0.0, 0.0, 0.0};

    double K15M[540] =
         {
           0.0, 7.0, 0.0,    0.0,  0.0,  0.0,-35.0,   0.0,   0.0,  0.0,      0.0,  0.0,        0.0,  0.0,     0.0,     21.0, 0.0,      0.0,   0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,       0.0, 0.0,  0.0, -1.0,  0.0,  0.0,    0.0,      0.0,      0.0,  0.0, 0.0,
           0.0, 0.0, 0.0,    0.0,  1.0,  0.0,  0.0,   0.0,   0.0,  0.0,      0.0, -10.0/3.0,   0.0,  0.0,     0.0,      0.0, 0.0,      0.0,   0.0,   0.0,  0.0,  0.0,  1.0,  0.0,  0.0,       0.0, 0.0,  0.0,  0.0,  0.0,  0.0,    0.0,      0.0,      0.0,  0.0, 0.0,
           0.0,-5.0, 0.0,    0.0,  0.0,  0.0,  5.0,   0.0,  60.0,  0.0,      0.0,  0.0,        0.0,  0.0,     0.0,      9.0, 0.0,   -120.0,   0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,       0.0, 0.0,  0.0, -1.0,  0.0, 12.0,    0.0,      0.0,      0.0,  0.0, 0.0,
           0.0, 0.0, 0.0,    0.0, -1.0,  0.0,  0.0,   0.0,   0.0,  0.0,      0.0,  0.0,        0.0, 10.0/3.0, 0.0,      0.0, 0.0,      0.0,   0.0,   0.0,  0.0,  0.0,  1.0,  0.0, -10.0/3.0,  0.0, 0.0,  0.0,  0.0,  0.0,    0.0,  0.0,      0.0,      0.0,  0.0, 0.0,
           0.0, 3.0, 0.0,    0.0,  0.0,  0.0,  5.0,   0.0, -60.0,  0.0,      0.0,  0.0,        0.0,  0.0,     0.0,      1.0, 0.0,    -40.0,   0.0,  80.0,  0.0,  0.0,  0.0,  0.0,  0.0,       0.0, 0.0,  0.0, -1.0,  0.0,  20.0,   0.0,    -80.0/3.0,  0.0,  0.0, 0.0,
           0.0, 0.0, 0.0,    0.0,  1.0,  0.0,  0.0,   0.0,   0.0,  0.0,      0.0, 2.0,         0.0, -16.0/3.0,0.0,      0.0, 0.0,      0.0,   0.0,   0.0,  0.0,  0.0,  1.0,  0.0, -16.0/3.0,  0.0, 3.2,  0.0,  0.0,  0.0,    0.0,  0.0,      0.0,      0.0,  0.0, 0.0,
           0.0,-1.0, 0.0,    0.0,  0.0,  0.0, -3.0,   0.0,  24.0,  0.0,      0.0,  0.0,        0.0,  0.0,     0.0,     -3.0, 0.0,     48.0,   0.0, -48.0,  0.0,  0.0,  0.0,  0.0,  0.0,       0.0, 0.0,  0.0, -1.0,  0.0,   24.0,  0.0,    -48.0,      0.0, 12.8, 0.0,
           0.0, 0.0,-2.1875, 0.0,  0.0,  0.0,  0.0, -6.5625, 0.0, 13.125,    0.0,  0.0,        0.0,  0.0,     0.0,      0.0,-6.5625,   0.0,  26.25,  0.0,-10.5,  0.0,  0.0,  0.0,  0.0,       0.0, 0.0,  0.0,  0.0, -2.1875, 0.0, 13.125,    0.0,     -10.5, 0.0, 1.0,
          -1.0, 0.0, 0.0,   -3.0,  0.0, 24.0,  0.0,  0.0,    0.0,  0.0,     -3.0,  0.0,       48.0,  0.0, -   48.0,     0.0, 0.0,      0.0,   0.0,   0.0,  0.0, -1.0,  0.0, 24.0,  0.0,     -48.0, 0.0, 12.8,  0.0,  0.0,    0.0,  0.0,      0.0,      0.0,  0.0, 0.0,
           0.0, 0.0, 1.0,    0.0,  0.0,  0.0,  0.0,  1.0,    0.0, -16.0/3.0, 0.0,  0.0,        0.0,  0.0,     0.0,      0.0,-1.0,      0.0,   0.0,   0.0,  3.2,  0.0,  0.0,  0.0,  0.0,       0.0, 0.0,  0.0,  0.0, -1.0,    0.0, 16.0/3.0,  0.0,     -3.2,  0.0, 0.0,
           1.0, 0.0, 0.0,   -1.0,  0.0,-20.0,  0.0,  0.0,    0.0,  0.0,     -5.0,  0.0,       40.0,  0.0,     80.0/3.0, 0.0, 0.0,      0.0,   0.0,   0.0,  0.0, -3.0,  0.0, 60.0,  0.0,     -80.0, 0.0,  0.0,  0.0,  0.0,    0.0,  0.0,      0.0,      0.0,  0.0, 0.0,
           0.0, 0.0,-1.0,    0.0,  0.0,  0.0,  0.0,  5.0,    0.0, 10.0/3.0,  0.0,  0.0,        0.0,  0.0,     0.0,      0.0, 5.0,      0.0, -20.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,       0.0, 0.0,  0.0,  0.0, -1.0,    0.0, 10.0/3.0,  0.0,      0.0,  0.0, 0.0,
          -1.0, 0.0, 0.0,    9.0,  0.0, 12.0,  0.0,  0.0,    0.0,  0.0,      5.0,  0.0,     -120.0,  0.0,     0.0,      0.0, 0.0,      0.0,   0.0,   0.0,  0.0, -5.0,  0.0, 60.0,  0.0,       0.0, 0.0,  0.0,  0.0,  0.0,    0.0,  0.0,      0.0,      0.0,  0.0, 0.0,
           0.0, 0.0, 1.0,    0.0,  0.0,  0.0,  0.0,-15.0,    0.0,  0.0,      0.0,  0.0,        0.0,  0.0,     0.0,      0.0,15.0,      0.0,   0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,       0.0, 0.0,  0.0,  0.0, -1.0,    0.0,  0.0,      0.0,      0.0,  0.0, 0.0,
           1.0, 0.0, 0.0,  -21.0,  0.0,  0.0,  0.0,  0.0,    0.0,  0.0,     35.0,  0.0,        0.0,  0.0,     0.0,      0.0, 0.0,      0.0,   0.0,   0.0,  0.0, -7.0,  0.0,  0.0,  0.0,       0.0, 0.0,  0.0,  0.0,  0.0,    0.0,  0.0,      0.0,      0.0,  0.0, 0.0};
    n_ao_d5=0;
    for(auto &b: s){
        for(auto &c: b.contr){
            n_ao_d5+=2*c.l+1;
//             fprintf(out_stream,"%d %d %d\n",n_ao_d5,2*c.l+1,c.l);
//             getchar();
        }
    }
//     fprintf(out_stream,"%d %d\n",n_ao,n_ao_d5);
    T_D5 = new double[n_ao*      n_ao_d5];
    V_D6 = new double[n_ao*(n_ao-n_ao_d5)];
    set_zero_matr(T_D5,n_ao*n_ao_d5);
    set_zero_matr(V_D6, n_ao*(n_ao-n_ao_d5));
    int i=0;
    int j=0;
    for(auto &b: s){
        for(auto &c: b.contr){
            if(c.l==0)T_D5[i*n_ao_d5+j]=1;
            if(c.l==1){
                T_D5[(i+0)*n_ao_d5+j+0]=1;
                T_D5[(i+1)*n_ao_d5+j+1]=1;
                T_D5[(i+2)*n_ao_d5+j+2]=1;
            }
            if(c.l==2){
                for(int ii=0;ii<6;ii++)
                for(int jj=0;jj<5;jj++)
                    T_D5[(i+ii)*n_ao_d5+j+jj]=D5M[jj*6+ii];
            }
            if(c.l==3){
//                 fprintf(out_stream,"d5 is not realized for f orbitals\n");
//                 exit(1);
                for(int ii=0;ii<10;ii++)
                for(int jj=0;jj<7 ;jj++)
                    T_D5[(i+ii)*n_ao_d5+j+jj]=F7M[jj*10+ii];
            }
            if(c.l==4){
//                 fprintf(out_stream,"%d %d\n",c.size(),2*c.l+1);
//                 fprintf(out_stream,"d5 is not realized for g orbitals\n");
//                 exit(1);
                for(int ii=0;ii<15;ii++)
                for(int jj=0;jj<9;jj++)
                    T_D5[(i+ii)*n_ao_d5+j+jj]=G9M[jj*15+ii];
            }
            if(c.l==5){
//                 fprintf(out_stream,"%d %d\n",c.size(),2*c.l+1);
//                 fprintf(out_stream,"d5 is not realized for h orbitals\n");
//                 exit(1);
                for(int ii=0;ii<21;ii++)
                for(int jj=0;jj<11;jj++)
                    T_D5[(i+ii)*n_ao_d5+j+jj]=H11M[jj*21+ii];
            }
            if(c.l==6){
                for(int ii=0;ii<28;ii++)
                for(int jj=0;jj<13;jj++)
                    T_D5[(i+ii)*n_ao_d5+j+jj]=I13M[jj*28+ii];
            }
            if(c.l==7){
                for(int ii=0;ii<36;ii++)
                for(int jj=0;jj<15;jj++)
                    T_D5[(i+ii)*n_ao_d5+j+jj]=K15M[jj*36+ii];
            }
            if(c.l>7){
                fprintf(out_stream,"%d %d\n",c.size(),2*c.l+1);
                fprintf(out_stream,"d5 is not realized for orbitals with L>7\n");
                exit(1);
            }
            
            i+=c.cartesian_size();
            j+=2*c.l+1;

        }
    }
    if(LINEAR)for(int i_r=0;i_r<S.n_rep;i_r++){
        S.n_ao_d5[i_r]=0;
        for(auto &b: s)
        for(auto &c: b.contr){
            if(c.l==0)if(i_r<1)S.n_ao_d5[i_r]++;
            if(c.l==1)if(i_r<3)S.n_ao_d5[i_r]++;
            if(c.l==2)if(i_r<5)S.n_ao_d5[i_r]++;
            if(c.l==3)if(i_r<7)S.n_ao_d5[i_r]++;
            if(c.l==4)if(i_r<9)S.n_ao_d5[i_r]++;
            if(c.l==5)         S.n_ao_d5[i_r]++;

        }
//         printf("A %d %d\n",i_r,S.n_ao_d5[i_r]);
        S.T_D5[i_r] = new double[ n_ao*n_ao];//excessive size to make it square
        S.P[i_r] = new double[ n_ao*n_ao];
        set_zero_matr(S.T_D5[i_r],n_ao*n_ao);//excessive size to make it square
        int i=0;
        int j=0;
        int jj;
        calc_S_AO();
        double * S_AO_f = new double[n_ao*n_ao];
        set_zero_matr(S_AO_f,n_ao*n_ao);
        
        for(auto &b: s){
            for(auto &c: b.contr){
                for(int k=0;k<c.cartesian_size();k++)
                for(int l=0;l<c.cartesian_size();l++)
                    S_AO_f[(i+k)*n_ao+i+l]=S_AO[(i+k)*n_ao+i+l];
                
                i+=c.cartesian_size();
            }
        }
        
        i=0;
        for(auto &b: s){
            for(auto &c: b.contr){
                if(c.l==0)if(i_r==0)S.T_D5[i_r][i*n_ao+j]=1;
                if(c.l==1){
                    if(i_r==0)S.T_D5[i_r][(i+2)*n_ao+j]=1;
                    if(i_r==1)S.T_D5[i_r][(i+0)*n_ao+j]=1;
                    if(i_r==2)S.T_D5[i_r][(i+1)*n_ao+j]=1;
                }
                if(c.l==2){
                    if(i_r==0)jj=2;
                    if(i_r==1)jj=3;
                    if(i_r==2)jj=1;
                    if(i_r==3)jj=0;
                    if(i_r==4)jj=4;
                    
                    if(i_r<5)for(int ii=0;ii<6;ii++)
                        S.T_D5[i_r][(i+ii)*n_ao+j]=D5M[jj*6+ii];
                }
                if(c.l==3){
                    if(i_r==0)jj=3;
                    if(i_r==1)jj=4;
                    if(i_r==2)jj=2;
                    if(i_r==3)jj=1;
                    if(i_r==4)jj=5;
                    if(i_r==5)jj=6;
                    if(i_r==6)jj=0;
                    
                    if(i_r<7)for(int ii=0;ii<10;ii++)
                        S.T_D5[i_r][(i+ii)*n_ao+j]=F7M[jj*10+ii];
                }
                if(c.l==4){
                    if(i_r==0)jj=4;
                    if(i_r==1)jj=5;
                    if(i_r==2)jj=3;
                    if(i_r==3)jj=2;
                    if(i_r==4)jj=6;
                    if(i_r==5)jj=7;
                    if(i_r==6)jj=1;
                    if(i_r==7)jj=0;
                    if(i_r==8)jj=8;
                    

                    if(i_r<9)for(int ii=0;ii<15;ii++)
                        S.T_D5[i_r][(i+ii)*n_ao+j]=G9M[jj*15+ii];
                }
                if(c.l==5){
                    if(i_r== 0)jj= 5;
                    if(i_r== 1)jj= 6;
                    if(i_r== 2)jj= 4;
                    if(i_r== 3)jj= 3;
                    if(i_r== 4)jj= 7;
                    if(i_r== 5)jj= 8;
                    if(i_r== 6)jj= 2;
                    if(i_r== 7)jj= 1;
                    if(i_r== 8)jj= 9;
                    if(i_r== 9)jj=10;
                    if(i_r==10)jj= 0;
                    
                    if(i_r<11)for(int ii=0;ii<21;ii++)
                        S.T_D5[i_r][(i+ii)*n_ao+j]=H11M[jj*21+ii];
                }
                if(c.l>5){
                    fprintf(out_stream,"%d %d\n",c.size(),2*c.l+1);
                    fprintf(out_stream,"d5 is not realized for i orbitals\n");
                    exit(1);
//                     for(int ii=0;ii<15;ii++)
//                     for(int jj=0;jj<9;jj++)
//                         T_D5[(i+ii)*n_ao_d5+j+jj]=G9M[ii*9+jj];
                }
                
                i+=c.cartesian_size();
                j++;
        
            }
        }
        
//         printf("B %d %d\n",i_r,S.n_ao_d5[i_r]);
        
//         printf("S:\n");PrintMatr(S_AO,n_ao,n_ao,1);
//         printf("S:\n");PrintMatr(S_AO_f,n_ao,n_ao,1);
//         printf("P:\n");PrintMatr(S.T_D5[i_r],n_ao,n_ao,1);
        transform_back(BUF,S_AO_f,S.T_D5[i_r],BUF+n_ao*n_ao,n_ao);
//         printf("S2:\n");PrintMatr(BUF,n_ao,n_ao,1);
        inv_matr_constr(BUF,n_ao);
//         printf("S2_inv:\n");PrintMatr(BUF,n_ao,n_ao,1);        
        transform(S.P[i_r],BUF,S.T_D5[i_r],BUF+n_ao*n_ao,n_ao);
//         memcpy(S.P[i_r],S.T_D5[i_r],sizeof(double)*n_ao*n_ao)
        memcpy(BUF,S.P[i_r],sizeof(double)*n_ao*n_ao);
//         printf("P1:\n");PrintMatr(BUF,n_ao,n_ao,1);        
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        n_ao,n_ao,n_ao,1.0,
                        BUF,n_ao,
                        S_AO_f,n_ao,0.0,
                        S.P[i_r],n_ao);
//         printf("P2:\n");PrintMatr(S.P[i_r],n_ao,n_ao,1);        
    
    }
//     if(LINEAR)for(int i_r=0;i_r<S.n_rep;i_r++){
//         printf("C %d %d\n",i_r,S.n_ao_d5[i_r]);
//     }
    
//     PrintMatr(T_D5, n_ao, n_ao_d5,0);
//     exit(0);
    
    
    return 1;
}


int transform_from_col_MO(double * M_O, double * M, int n, double * V1, int n1, double * V2, int n2){
    
    //transformes AO matrix to MO matrix with V1 and V2 if orbitals are written as columns
    //M_O = V1^(t)*M*V2
    
    double *BUF;
    BUF = new double[n*n2];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                         n,n2,n,1.0,
                         M,n,
                         V2,n2,0.0,
                         BUF,n2);
     
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                         n1,n2,n,1.0,
                         V1,n1, 
                         BUF,n2,0.0,
                         M_O,n2);
    
//     PrintMatr(M_O,n1,n2,1);
    
    delete[] BUF;
    
    return 1;
}

int transform_from_col_MO_compl(double * M_O, double * M, int n, double * V1, int n1, double * V2, int n2){
    
    //transformes AO matrix to MO matrix with V1 and V2 if orbitals are written as columns
    //M_O = V1^(t)*M*V2
    
    double * Mi   = M   + n *n;
    double * V1i  = V1  + n1*n ;
    double * V2i  = V2  + n *n2;
    double * M_Oi = M_O + n1*n2;
    
    
    double *BUFr;
    double *BUFi;
    BUFr = new double[n*n2];
    BUFi = new double[n*n2];
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                         n,n2,n,1.0,
                         M,n,
                         V2,n2,0.0,
                         BUFr,n2);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                         n,n2,n,-1.0,
                         Mi,n,
                         V2i,n2,1.0,
                         BUFr,n2);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                         n,n2,n,1.0,
                         M,n,
                         V2i,n2,0.0,
                         BUFi,n2);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                         n,n2,n,1.0,
                         Mi,n,
                         V2,n2,1.0,
                         BUFi,n2);
     
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                         n1,n2,n,1.0,
                         V1,n1, 
                         BUFr,n2,0.0,
                         M_O,n2);
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                         n1,n2,n,1.0,
                         V1i,n1, 
                         BUFi,n2,1.0,
                         M_O,n2);
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                         n1,n2,n,1.0,
                         V1,n1, 
                         BUFi,n2,0.0,
                         M_Oi,n2);
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                         n1,n2,n,-1.0,
                         V1i,n1, 
                         BUFr,n2,1.0,
                         M_Oi,n2);
//     PrintMatr(M_O,n1,n2,1);
    
    delete[] BUFr;
    delete[] BUFi;
    
    return 1;
}



int change_orbs(double * V, int i, int j, int dim){
    
    double buf;
    
    for(int k=0;k<dim;k++){
        buf        = V[i*dim+k];
        V[i*dim+k] = V[j*dim+k];
        V[j*dim+k] = buf;
    }
    
    return 0;
    
}

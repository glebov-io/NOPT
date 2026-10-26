# include <stdlib.h>
#include <math.h>

#include "common_vars.h"
#include "etc.h"
#include "z_matrix.h"
#include "matr.h"
#include "geom.h"

Z_matrix::Z_matrix(){
    
    n = nullptr;
    p = nullptr;
    dP= nullptr;
}


Z_matrix::~Z_matrix(){
    if(n != nullptr) delete[] n;
    if(p != nullptr) delete[] p;
    if(dP!= nullptr) delete[]dP;
    
}

Z_matrix& Z_matrix::operator= (Z_matrix& other){
    init(other.n_at);
    
    for(int i=0; i<3*n_at-6;i++) n[i]=other.n [i];
    for(int i=0; i<3*n_at-6;i++) p[i]=other.p [i];
    for(int i=0; i<3*n_at-6;i++)dP[i]=other.dP[i];
    
    for(int i=0; i<9;i++)xyz_first[i]=other.xyz_first[i];
    
    return *this;
}


int Z_matrix::init(int ext_n_at){
    
    
    if(n != nullptr)delete[]n ;
    if(p != nullptr)delete[]p ;
    if(dP!= nullptr)delete[]dP;
    
    
    // fewer than 3 atoms: no internal-coordinate Z-matrix (the Cartesian input does not need one; Z-matrix driven scans and the Z-matrix numerical
    // optimizer are unavailable). A diatomic used to stop the program here.
    n_at = ext_n_at;
    if(ext_n_at<3){
        printf("WARNING: no Z-matrix for a molecule with %d atom(s) (Cartesian coordinates are used as given)\n", ext_n_at);
        n = nullptr;
        p = nullptr;
        dP= nullptr;

        return 1;
    }
    
    
    
    n = new int   [3*n_at-6];
    p = new double[3*n_at-6];
    dP= new double[3*n_at-6];
    
    n[0] = 0;
    n[1] = 1;
    n[2] = 0;
    
    for(int i=3;i<n_at;i++){
        n[3*i-6]=i-1;
        n[3*i-5]=i-2;
        n[3*i-4]=i-3;
    }
    
    
    return 0;
}

inline double r(double * xyz, int n, int m){
    
    return dist(xyz[3*n],xyz[3*n+1],xyz[3*n+2],
                xyz[3*m],xyz[3*m+1],xyz[3*m+2],1);
    
}

inline double a(double * xyz, int n, int m, int k){
    
    return angle(xyz[3*n],xyz[3*n+1],xyz[3*n+2],
                 xyz[3*m],xyz[3*m+1],xyz[3*m+2],
                 xyz[3*k],xyz[3*k+1],xyz[3*k+2],1);
    
}

inline double d(double * xyz, int n, int m, int k, int l){
    
    return dihedral(xyz[3*n],xyz[3*n+1],xyz[3*n+2],
                    xyz[3*m],xyz[3*m+1],xyz[3*m+2],
                    xyz[3*k],xyz[3*k+1],xyz[3*k+2],
                    xyz[3*l],xyz[3*l+1],xyz[3*l+2],1);
    
}


int Z_matrix::calc_Z(double * xyz){
    
    if(n_at<3) return 0;
    
    p[0] = r(xyz,1,n[0]);//why 1 bu not n[1]??
    p[1] = r(xyz,2,n[1]);
    p[2] = a(xyz,2,n[1],n[0]);
    
    for(int i=3;i<n_at;i++){
        p[3*i-6]=r(xyz,i,n[3*i-6]);
        p[3*i-5]=a(xyz,i,n[3*i-6],n[3*i-5]);
        p[3*i-4]=d(xyz,i,n[3*i-6],n[3*i-5],n[3*i-4]);
    }                         
    
    for(int i=0;i<9;i++)xyz_first[i]=xyz[i];
    
    return 0;
}

inline int add_shitfted_atom(double * R, int n, int m, double x, double y, double z){
    
    R[3*n+0] = R[3*m+0] + x;
    R[3*n+1] = R[3*m+1] + y;
    R[3*n+2] = R[3*m+2] + z;
    
    return 0;
}


int add_atom(double * R, int n, int m, int k, int l, double r, double a, double d, double * e){
    
    e[0]=R[3*m+0]-R[3*k+0];
    e[1]=R[3*m+1]-R[3*k+1];
    e[2]=R[3*m+2]-R[3*k+2];
    
    make_norm(e,0,3);
    
    e[3]=R[3*k+0]-R[3*l+0];
    e[4]=R[3*k+1]-R[3*l+1];
    e[5]=R[3*k+2]-R[3*l+2];
    
    double s =e[0]*e[3]+e[1]*e[4]+e[2]*e[5];
    
    e[3]=e[3]-s*e[0];
    e[4]=e[4]-s*e[1];
    e[5]=e[5]-s*e[2];
    
    make_norm(e,1,3);
    
    e[6]=e[1]*e[5]-e[2]*e[4];
    e[7]=e[2]*e[3]-e[0]*e[5];
    e[8]=e[0]*e[4]-e[1]*e[3];
    
    make_norm(e,2,3);
    
    double x = -cos(a);
    double y = -sin(a)*cos(d);
    double z = -sin(a)*sin(d);
    
    add_shitfted_atom(R, n, m, 
                      (x*e[0]+y*e[3]+z*e[6])*r, 
                      (x*e[1]+y*e[4]+z*e[7])*r, 
                      (x*e[2]+y*e[5]+z*e[8])*r);
    
    
    
    return 0;
}



int Z_matrix::calc_xyz(double * xyz){
    
    double x,y,z;
    double a;
    set_zero_matr(xyz,3*n_at);
    
    double e[9];
    
    xyz[0]=xyz_first[0];
    xyz[1]=xyz_first[1];
    xyz[2]=xyz_first[2];
    double l =(xyz_first[0]-xyz_first[3])*(xyz_first[0]-xyz_first[3])+
              (xyz_first[1]-xyz_first[4])*(xyz_first[1]-xyz_first[4])+
              (xyz_first[2]-xyz_first[5])*(xyz_first[2]-xyz_first[5]);
    l=sqrt(l);
    
    xyz[3]=xyz_first[0]+(xyz_first[3]-xyz_first[0])*p[0]/l;
    xyz[4]=xyz_first[1]+(xyz_first[4]-xyz_first[1])*p[0]/l;
    xyz[5]=xyz_first[2]+(xyz_first[5]-xyz_first[2])*p[0]/l;
    
    xyz[ 9]=xyz_first[6];//tmp atom - copy of atom 2
    xyz[10]=xyz_first[7];//tmp atom - copy of atom 2
    xyz[11]=xyz_first[8];//tmp atom - copy of atom 2
    add_atom(xyz,2,1,0,3,p[1],p[2],0,e);

    
    
    
//     add_shitfted_atom(xyz, 2, n[1], x, y, z);
    
//     add_atom(xyz,3,2,1,0,p[3],p[4],p[5],e);
    
    for(int i=3;i<n_at;i++)
        add_atom(xyz,i,n[3*i-6],n[3*i-5],n[3*i-4]
                      ,p[3*i-6],p[3*i-5],p[3*i-4],e);
//         p[3*i-6]=r(xyz,i,n[3*i-6]);
//         p[3*i-5]=a(xyz,i,n[3*i-6],n[3*i-5]);
//         p[3*i-4]=d(xyz,i,n[3*i-6],n[3*i-5],n[3*i-4]);
//     }                         
    
    for(int i=0;i<9;i++)xyz_first[i]=xyz[i];//update xyz_first
    
    return 0;
}


int Z_matrix::print(){
    
    fprintf(out_stream,"r(%d,%d) = %.4f\n",   1,n[0],p[0]/1.889725989);
    fprintf(out_stream,"r(%d,%d) = %.4f\n",   2,n[1],p[1]/1.889725989);
    fprintf(out_stream,"a(%d,%d,%d) = %.4f\n",2,n[1],n[0],p[2]/M_PI*180);
    
    for(int i=3;i<n_at;i++){
        fprintf(out_stream,"r(%d,%d) = %.4f\n",      i,n[3*i-6],                  p[3*i-6]/1.889725989);
        fprintf(out_stream,"a(%d,%d,%d) = %.4f\n",   i,n[3*i-6],n[3*i-5],         p[3*i-5]/M_PI*180);
        fprintf(out_stream,"d(%d,%d,%d,%d) = %.4f\n",i,n[3*i-6],n[3*i-5],n[3*i-4],p[3*i-4]/M_PI*180);
    }                         
    
    
    return 0;
}

int Z_matrix::fprint(const char * name){
    
    FILE * out_file = fopen(name,"w");
    FILE * backup;
    backup =out_stream;
    out_stream = out_file;
    print();
    out_stream = backup;
    fclose(out_file);
    
    return 0;
}



int Z_matrix::set_deriv_step(double dR, double dA, double dD){
    
//     if(dP==nullptr)dP= new double[3*n_at-6];
    
    dP[0] = dR;
    dP[1] = dR;
    dP[2] = dA;
    
    for(int i=3;i<n_at;i++){
        dP[3*i-6]=dR;
        dP[3*i-5]=dA;
        dP[3*i-4]=dD;
    }                         
    
    
    return 0;
}

int z_mat_read_atom_err(int a, int b){
    
    printf("ERROR: atom %d can not be connected  with atom %d\n", b,a);
    exit(1);
    return 0;
}

int find_int_and_move(char* l, int * a){
    
    int r=0;
//     printf("%s\n",l);
    sscanf(l,"%d",a);
    while(l[r]==' ')r++;
    while(isdigit(l[r]))r++;
//     printf("%d\n",a[0]);
//     printf("%s\n",l+r);
//     getchar();
    
    return r;
}


int Z_matrix::read_line(char * inp_line, int i){
    
    if(i<0){
        printf("unexpected error in Z_matrix::read(char * inp_line, int i): i<0\n");
        exit(1);
    }
    
    if(i==0) return 0;
    
    
    int r=0;
    while(inp_line[r]==' ')r++;
    if(i==1){
        r=find_int_and_move(inp_line,n);
        read_coord(inp_line+r,p,0,0);
        if(n[0]>i) z_mat_read_atom_err(n[0],i);
        return 0;
    }
    
    if(i==2){
        r =find_int_and_move(inp_line  ,n+1);
        r+=read_coord       (inp_line+r,p,1,0);
        r+=find_int_and_move(inp_line+r,n+2);
        r+=read_coord       (inp_line+r,p,2,1);
        
        if(n[1]>i) z_mat_read_atom_err(n[1],i);
        if(n[2]>i) z_mat_read_atom_err(n[2],i);
        return 0;
    }
    
    r =find_int_and_move(inp_line  ,n+3*i-6);
    r+=read_coord       (inp_line+r,p,3*i-6,0);
    r+=find_int_and_move(inp_line+r,n+3*i-5);
    r+=read_coord       (inp_line+r,p,3*i-5,1);
    r+=find_int_and_move(inp_line+r,n+3*i-4);
    r+=read_coord       (inp_line+r,p,3*i-4,1);
    
    if(n[3*i-6]>i) z_mat_read_atom_err(n[3*i-6],i);
    if(n[3*i-5]>i) z_mat_read_atom_err(n[3*i-5],i);
    if(n[3*i-4]>i) z_mat_read_atom_err(n[3*i-4],i);
        
    return 0;
}

int Z_matrix::read_xyz(char * inp_line, int i){
    
//     printf("read_xyz is not written\n");
//     printf("%s\n",inp_line);
    int n;
    sscanf(inp_line,"%d %lf %lf %lf", &n, xyz_first+3*i, xyz_first+3*i+1, xyz_first+3*i+2);
    if(n!=i){
        printf("ERROR: could not read atom %d from line %d of xyz-3 group\n",n,i);
        printf("       option is not yet supported\n");
        exit(0);
    }
    
    
    return 0;
}





int Z_matrix::scale_R(double au_coef){
    
    if(n_at<3)return 0;
    
    p[0] = p[0]*au_coef;
    p[1] = p[1]*au_coef;
    
    for(int i=3;i<n_at;i++){
        p[3*i-6]=p[3*i-6]*au_coef;
    }                         
    
    
    return 0;
}



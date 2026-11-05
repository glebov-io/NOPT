#ifndef Z_MAT_H
#define Z_MAT_H

class Z_matrix;

class Z_matrix 
{
    public:
        
        int n_at;
        int * n;
        double * p;
        double * dP;
        int n_xyz;
        double xyz_first[9];
        
        
        Z_matrix();
        ~Z_matrix();
        int init(int ext_n_at);
        Z_matrix& operator= (Z_matrix& other);
        int calc_Z  (double * xyz);
        int calc_xyz(double * xyz);
        int print();
        int fprint(const char * name);
        int set_deriv_step(double dR, double dA, double dD);
        int read_line(char * inp_line, int i);
        int read_xyz(char * inp_line, int i);
        int scale_R(double au_coef);
//         int reorder();
};

#endif

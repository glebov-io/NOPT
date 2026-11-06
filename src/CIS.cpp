//standart

//user
# include "blas_link.h"
# include "energy_result.h"
# include "molecule.h"
# include "timer.h"
# include "defaults.h"
# include "doCI_data.h"
# include "matr.h"
# include "SCF.h"
# include "CIS.h"
# include "common_vars.h"
# include "RI.h"
# include "davidson.h"


extern long aux_n_ao;     //?
extern double* RI_AA;     //?
extern double* AUX_Vm05;  //?

//------------------------------------------------------------------------------------------------------------------------
CIS_engine::CIS_engine()
{
  DM    = NULL;
  B     = NULL;
  F     = NULL;
  H_CIS = NULL;  //новое
  A1    = NULL;
  C     = NULL;
  d     = NULL;
  E_app = NULL;
  E_p   = NULL;
  H_p   = NULL;
  nums  = NULL;
}

double * d_T;
double * CI;

//------------------------------------------------------------------------------------------------------------------------
CIS_engine::~CIS_engine()
{
  if (DM    != NULL) delete[] DM;
  if (B     != NULL) delete[] B;
  if (F     != NULL) delete[] F;
  if (H_CIS != NULL) delete[] H_CIS;  //новое
  if (A1    != NULL) delete[] A1;
  if (C     != NULL) delete[] C;
  if (d     != NULL) delete[] d;
  if (E_app != NULL) delete[] E_app;
  if (E_p   != NULL) delete[] E_p;
  if (H_p   != NULL) delete[] H_p;
  if (nums  != NULL) delete[] nums;
}

//------------------------------------------------------------------------------------------------------------------------
inline double two_el_int(const double* B, long n_ao, long n_aux, int p, int q, int r, int s)
{
  return cblas_ddot(n_aux, B + (p*n_ao + q)*n_aux, 1, B + (r*n_ao + s)*n_aux, 1);
}

//------------------------------------------------------------------------------------------------------------------------
int CIS_engine::calc_H_CIS()
{

  if (method==1)
  {  
    #pragma omp parallel for
    for (int ai=0; ai<n_v*n_c; ai++)
    {
      int a = ai/n_c;
      int i = ai%n_c;

      for (int j=0; j<n_c; j++)
      {
        for (int b=0; b<n_v; b++)
        {  
	        double H_iajb = 0;

          int i1 = i + n_f;           //
          int j1 = j + n_f;           //  поправка на нумерацию
          int a1 = a + n_f + n_c;     //
          int b1 = b + n_f + n_c;     //
             
          if ( (a==b)&&(i==j) ) { H_iajb = C1 + F[a1*n_ao+a1] - F[i1*n_ao+i1]; }
          else if (a==b) { H_iajb = -F[i1*n_ao+j1]; }
          else if (i==j) { H_iajb = F[a1*n_ao+b1]; }
              
          H_iajb += 2*two_el_int(B, n_ao, n_aux, a1, i1, b1, j1) - two_el_int(B, n_ao, n_aux, a1, b1, i1, j1);
              
          H_CIS[((i*n_v + a)*n_c + j)*n_v + b] = H_iajb;            
        }
      }    
    }
  }

  return 0;
}

//------------------------------------------------------------------------------------------------------------------------
int CIS_engine::H_mult(int n_s_mult)
{
  set_zero_matr(d, n_s_mult * n_c * n_v);
  if (n_s_mult > n_b) {printf("Ошибка! n_s не может быть больше, чем n_b !"); exit(1);}

  if (method==1)
  {
    /*#pragma omp parallel for
    for (int nj=0; nj<n_s_mult*n_c; nj++)
    {
      int n=nj/n_c;
      int j=nj%n_c;

      for (int b=0; b<n_v; b++)
      {
        for (int i=0; i<n_c; i++)
        {
          for (int a=0; a<n_v; a++)
          {
            d[(n*n_c + j)*n_v + b] += C[(n*n_c + i)*n_v + a] * H_CIS[((j*n_v + b)*n_c + i)*n_v + a];
          }
        }
      }        
    }*/
  
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_s_mult, n_c*n_v, n_c*n_v, 1,//
    C, n_c*n_v, H_CIS, n_c*n_v, 0.0, d, n_c*n_v);
  
  }

  else
  {
    set_zero_matr(d_T, n_s_mult * n_c * n_v);
    set_zero_matr(CI, n_b * n_aux);

    #pragma omp parallel for
    for (int n=0; n<n_s_mult; n++)
    {
      for (int p=0; p<n_c; p++)
      {
        for (int q=0; q<n_v; q++)
        {
          for (int I=0; I<n_aux; I++)  
          {
            CI[n*n_aux + I] += C[(n*n_c + p)*n_v + q] * 2 * B[((p + n_f)*n_ao + (q + n_f + n_c))*n_aux + I];
          }
        }
      }        
    }

    #pragma omp parallel for
    for (int n=0; n<n_s_mult; n++)
    {
      for (int I=0; I<n_aux; I++)
      {
        for (int r=0; r<n_c; r++)
        {
          for (int s=0; s<n_v; s++)
          {
            d[(n*n_c + r)*n_v + s] += CI[n*n_aux + I] * B[((r + n_f)*n_ao + (s + n_f + n_c))*n_aux + I];
          }
        }
      }  
    }

    #pragma omp parallel for
    for (int jb=0; jb<n_c*n_v; jb++)   //второй цикл
    {
      int j = jb/n_v;
      int b = jb%n_v;

      for (int i=0; i<n_c; i++)
      {
        for (int a=0; a<n_v; a++)
        {            
          int i1 = i + n_f;           //
          int j1 = j + n_f;           //  поправка на нумерацию
          int a1 = a + n_f + n_c;     //
          int b1 = b + n_f + n_c;     //

          double H_iajb = 0;
              
          if ( (a==b)&&(i==j) ) { H_iajb = C1 + F[a1*n_ao+a1] - F[i1*n_ao+i1]; }
          else if (a==b) { H_iajb = -F[i1*n_ao+j1]; }
          else if (i==j) { H_iajb = F[a1*n_ao+b1]; }
              
          H_iajb += - two_el_int(B, n_ao, n_aux, a1, b1, i1, j1);

	        for (int n=0; n<n_s_mult; n++)
          {
	          //d[(n*n_c + j)*n_v + b] += C[(n*n_c + i)*n_v + a] * H_iajb;
            d_T[(j*n_v + b)*n_s_mult + n] += C[(n*n_c + i)*n_v + a] * H_iajb;
          }
        }
      }
    }

    #pragma omp parallel for    
    for (int n=0; n<n_s_mult; n++) //транспонирование d_T обратно в d
    {
      for (int jb=0; jb<n_c*n_v; jb++)
      {        
        d[n*n_c*n_v + jb] += d_T[jb*n_s_mult + n];    
      } 
    }

  }
    
  return 0;
}

//------------------------------------------------------------------------------------------------------------------------
int CIS_engine::set_par(molecule* ext_M, int ext_method, int ext_n_f, long ext_n_c, long ext_n_v, long ext_n_aux, int ext_n_s, int ext_n_b)
{    
  M=ext_M;
  M->mc=0;//set multi-configuration
  n_ao = M->n_ao;
  H = M->H_AO;
  DM = new double [n_ao*n_ao];

  method = ext_method;    
  n_f = ext_n_f;
  n_c = ext_n_c;
  n_v = ext_n_v;
  n_aux = ext_n_aux;
  n_s = ext_n_s;
  n_b = ext_n_b;
  
  B = new double [long(n_ao*n_ao*n_aux)];
  F = new double [n_ao * n_ao];
  
  A1 = new double [n_b * n_c * n_v];
  C  = new double [n_b * n_c * n_v];
  CI  = new double [n_b * n_aux];
  d  = new double [n_b * n_c * n_v];
  d_T  = new double [n_b * n_c * n_v];
  E_app = new double [n_c * n_v];
  E_p = new double [n_b];
  H_p = new double [n_b * n_b];
  nums = new int [n_b];
  
  if (method==1) H_CIS = new double[long(n_v*n_v*n_c*n_c)];                               

  return 0;  
}

//------------------------------------------------------------------------------------------------------------------------
int CIS_engine::calc_MO_INTS(){

  double* A = new double[n_ao*n_ao*n_aux];
  set_zero_matr(A, n_ao*n_ao*n_aux);
  set_zero_matr(B, n_ao*n_ao*n_aux);
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_ao, n_ao*n_aux, n_ao,//
  1, M->MO_VEC, n_ao, RI_AA, n_ao*n_aux, 0.0, A, n_ao*n_aux);
  
  #pragma omp parallel for
  for (int i=0; i<n_ao; i++)
  {
    for (int j=0; j<n_ao; j++)
    {
      for (int k=0; k<n_aux; k++)
      {
        B[(j*n_ao + i)*n_aux + k] = A[(i*n_ao + j)*n_aux + k];
      }
    }
  }
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_ao, n_ao*n_aux, n_ao,//
  1, M->MO_VEC, n_ao, B, n_ao*n_aux, 0.0, A, n_ao*n_aux);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_ao*n_ao, n_aux, n_aux, 1, A, n_aux, AUX_Vm05, n_aux, 0.0, B, n_aux);
  
  delete[] A;
  
  return 0;
}

//------------------------------------------------------------------------------------------------------------------------
int CIS_engine::calc_F()
{
  
  printf_timer("Before F calculation");
  gen_HF_DM(DM, M->MO_VEC, n_ao, n_c + n_f);
  M->calc_F_AO(F, DM, 1.0);
  
  doCI_data D;
  D.first_alloc(M,M);
  D.AO_to_MO(F);
  printf_timer("After F calculation");
  
  D.AO_to_MO(H);
  
  return 0;
}

//------------------------------------------------------------------------------------------------------------------------
int CIS_engine::gen_bf()
{
  //double trace = 0;           // 2*<Ф0|H1|Ф0> = 2*trace(H)
  //double C = 0;               // <Ф0|H2|Ф0> = summ_(i,j э core)[ 2(i,i|j,j) - (i,j|i,j) ]
  C1 = M->V_nuc;                // C1 = M->V_nuc + trace + C
  for (int i=0; i < (n_f + n_c); i++)
  {
    C1 += F[i*n_ao+i] + H[i*n_ao+i];
  }
    
  set_zero_matr(A1, n_b * n_c * n_v);
  
  for (int i=0; i<n_c; i++)
  {
    for (int a=0; a<n_v; a++)                 //Считаем отдельно энергии для диагональных элементов
    {
      int i1 = i + n_f;           //  поправка на нумерацию
      int a1 = a + n_f + n_c;     //
      
      E_app[i*n_v + a] = C1 + F[a1*n_ao+a1] - F[i1*n_ao+i1]
      + 2*two_el_int(B, n_ao, n_aux, a1, i1, a1, i1) - two_el_int(B, n_ao, n_aux, a1, a1, i1, i1);
    }
  }
  
  find_N_min_els(nums, n_b, E_app, n_c*n_v);    //Сортируем эти энергии и выбираем n_b наименьших
  /*for (int i=0; i<n_b; i++)
  {
    E_p[i] = E_app[nums[i]];
  }*/
  
  for (int i=0; i<n_b; i++)
  {
    A1[i*n_c*n_v + nums[i]] = 1.0;
  }
  
  return 0;
}

//------------------------------------------------------------------------------------------------------------------------
int CIS_engine::gen_H() //генерация стартового гамильтониана gen_H
{  
  #pragma omp parallel for
  for (int I1=0; I1<n_b; I1++)
  {
    for (int J1=0; J1<n_b; J1++)
    {
      int i1 = nums[I1]/n_v + n_f;
      int a1 = nums[I1]%n_v + n_f + n_c;             
      int j1 = nums[J1]/n_v + n_f;
      int b1 = nums[J1]%n_v + n_f + n_c; 
      
      double H_iajb = 0;
              
      if ( (a1==b1)&&(i1==j1) ) { H_iajb = C1 + F[a1*n_ao+a1] - F[i1*n_ao+i1]; }
      else if (a1==b1) { H_iajb = -F[i1*n_ao+j1]; }
      else if (i1==j1) { H_iajb = F[a1*n_ao+b1]; }
              
      H_iajb += 2*two_el_int(B, n_ao, n_aux, a1, i1, b1, j1) - two_el_int(B, n_ao, n_aux, a1, b1, i1, j1);
      
      H_p[I1*n_b + J1] = H_iajb;
      
      //printf("H_p[%d;%d] = % .8e\n", I1, J1, H_p[I1*n_b + J1]); //вывод гамильтониана              
    }
  }
  
  return 0;
}

//------------------------------------------------------------------------------------------------------------------------
int CIS_engine::calc_evec()
{  
  lapack_diag(H_p, E_p, n_b);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_b, n_c*n_v, n_b, 1, H_p, n_b, A1, n_c*n_v, 0.0, C, n_c*n_v);
  
  return 0;
}

//------------------------------------------------------------------------------------------------------------------------
int CIS_engine::print_excited_states_info()
{
  printf("\n\n\n");
  printf(" _______________________________________\n");
  printf("|                                       |\n");
  printf("|    Information about excited states   |\n");
  printf("|_______________________________________|\n");

  energy_result_record("cis","e_app",-1,E_app,n_s);
  for (int n=0; n < n_s; n++)
  {
    printf("\n\n");
    printf("  Excited state %d     ENERGY = % 18.10f Eh\n", n+1, E_app[n]);
    printf(" _______________________________________\n");
    printf("| Single excitation |                   |\n");
    printf("|___________________|  SAP coefficient  |\n");
    printf("| From MO |  To MO  |                   |\n");
    printf("|_________|_________|___________________|\n");

    //find_max_coef_in_row(массив для номеров, сколько эл-тов искать, массив в котором искать, сколько элементов
    // сравнивать (т.е. длина строки), номер строки)
    find_max_coef_in_row(nums, 10, C, n_c*n_v, n);
    for (int k=0; k < 10; k++)
    { 
      int i = nums[k]/n_v;
      int a = nums[k]%n_v;
      if (fabs(C[(n*n_c + i)*n_v + a]) > 0.05)
      {
        printf("|    %-5d|    %-5d|   % .10f   |\n", i + n_f + 1, a + n_c + n_f + 1, C[(n*n_c + i)*n_v + a]);
      }
    }
    printf("|_________|_________|___________________|\n");
  }

  return 0;
}

//------------------------------------------------------------------------------------------------------------------------
int CIS_engine::gen_NO(std::vector <double> w_state, char * job_name)
{
  double * D_1 = new double[n_c*n_c];
  double * D_2 = new double[n_v*n_v];      
  
  int n_s_av =w_state.size();
  
  double sum_w_state = 0;      // нормировка весов состояний
  for (int i = 0; i < n_s_av; i++)
  {
    sum_w_state += w_state[i];
  }
  for (int i = 0; i < n_s_av; i++)
  {
    w_state[i] = w_state[i]/sum_w_state;
  }
  
  for (int i = 0; i < n_c; i++)   //считаем матрицу D_core для занятых орбиталей
  {
    for (int j = 0; j < n_c; j++)
    {      
      if (i == j) {D_1[n_c*i + j] = 2;}
      else {D_1[n_c*i + j] = 0;}
      for (int n = 0; n < n_s_av; n++)
      {
        for (int a = 0; a < n_v; a++)       //D_ij = 2*\delta_{ij} - SUMM(по a) (C_i^a)*(C_j^a)
        {
          D_1[n_c*i + j] += - w_state[n] * C[(n*n_c + i)*n_v + a] * C[(n*n_c + j)*n_v + a];
        }   
        //fprintf(OUT, "D_1[%d;%d] = % .8e\n", i, j, D_1[i*m + j]);
      }
    }
  }
  //fprintf(OUT, "\n\n\n");
  
  //считаем матрицу D_virt для свободных орбиталей, меняя при этом её знак (чтобы обратить порядок собственных чисел)
  for (int a = 0; a < n_v; a++)
  {
    for (int b = 0; b < n_v; b++)
    {
      D_2[n_v*a + b] = 0;
      for (int n = 0; n < n_s_av; n++)
      {
        for (int i = 0; i < n_c; i++)       //D_ab = SUMM(по i) (C_i^a)*(C_i^b)
        {
          D_2[n_v*a + b] += w_state[n] * C[(n*n_c + i)*n_v + a] * C[(n*n_c + i)*n_v + b];
        }
        //fprintf(OUT, "D_2[%d;%d] = % .8e\n", a, b, D_2[a*n + b]);
      }
    }
  }

  double E_1[n_c], E_2[n_v];
  
  lapack_diag(D_1, E_1, n_c);
  lapack_diag(D_2, E_2, n_v);
  
  

  for (int i = 0; i < n_f; i++) //передача заселенности орбиталей
  {
    M->orb_energy[i] = 2;
  }
  for (int i = 0; i < n_c; i++) 
  {
    M->orb_energy[i + n_f] = E_1[i];
  }
  for (int i = 0; i < n_v; i++) 
  {
    M->orb_energy[i + n_f + n_c] = E_2[i];
  }

  double B_1[n_c*n_ao];
  double B_2[n_v*n_ao];
  set_zero_matr(B_1, n_c*n_ao);
  set_zero_matr(B_2, n_v*n_ao);

  cblas_dcopy(n_c*n_ao, M->MO_VEC + n_f*n_ao, 1, B_1, 1);
  cblas_dcopy(n_v*n_ao, M->MO_VEC + (n_f + n_c)*n_ao, 1, B_2, 1);
  
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,//
  n_c, n_ao, n_c, 1, D_1, n_c, B_1, n_ao, 0.0, M->MO_VEC + n_f*n_ao, n_ao);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,//
  n_v, n_ao, n_v, 1, D_2, n_v, B_2, n_ao, 0.0, M->MO_VEC + (n_f + n_c)*n_ao, n_ao);
  
  M->sort_orbs('d');

  M->MO_gamess_format();

  
  char * name = new char[BUF_LINE_LENGTH];
  
  fprintf(out_stream,"\n\n");
  fprintf(out_stream,"Writing CIS natural orbitals:\n");

  sprintf(name,"%s_cis_nat_orb.out\0",job_name);
  M->GAMESS_type_out_print(name, -1);
  fprintf(out_stream,"visualization file: %s\n",name);
  
  sprintf(name,"%s_cis_nat_orb.orb\0",job_name);
  M->MO_print(name);
  fprintf(out_stream,"data file         : %s\n",name);

  //printf(" Орбитали напечатаны \n");
  
  

  delete[] name;
  
  return 0;
}

//------------------------------------------------------------------------------------------------------------------------
int CIS(molecule * M, cis_par * cis, char * job_name)
{
  cis->write_info();
  
  if(RI)gen_RI_AA(M);
  
  CIS_engine cis_engine;
  
  cis_engine.set_par(M, cis->method, cis->n_f, M->n_el_calc/2 - cis->n_f, M->n_mo - M->n_el_calc/2, aux_n_ao, cis->n_s, cis->dav.n_bf);

  cis_engine.calc_MO_INTS();
  
  cis_engine.calc_F();
  
  davidson_solver D;
  D.set_par_cis(&cis_engine,cis->dav);
  D.run_cis(1,1);

  cis_engine.print_excited_states_info();
  if (cis->nat_orb) {cis_engine.gen_NO(cis->w_state, job_name);}  
  
  fprintf(out_stream,"\n");
  fprintf(out_stream,"\n");
  fprintf(out_stream,"_______________________________________________________________________\n");
  fprintf(out_stream,"\n");
  fprintf(out_stream,"\n");
  printf_timer("CIS");
    

  return 0;
}

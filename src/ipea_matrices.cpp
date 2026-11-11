# include "matr.h"
# include "ipea_matrices.h"

void ipea_matrices(int n, const double * g1, const double * g2,
                   const double * gamma, const double * GAMMA,
                   double * U_IP, double * H_IP, double * U_EA, double * H_EA){

    //U_IP
    for(int i=0;i<n*n;i++)U_IP[i]=gamma[i]*0.5;


    //U_EA
    for(int i=0;i<n*n;i++)U_EA[i]=-U_IP[i];
    for(int i=0;i<n;i++)U_EA[i*(n+1)]=1.0+U_EA[i*(n+1)];


    //H_IP
    set_zero_matr(H_IP,n*n);

    for(int t=0;t<n;t++)
    for(int u=0;u<n;u++)
    for(int w=0;w<n;w++)
        H_IP[t*n+u]+= U_IP[t*n+w]*g1[u*n+w];//restricted variant


    for(int t=0;t<n;t++)
    for(int u=0;u<n;u++)
    for(int v=0;v<n;v++)
    for(int x=0;x<n;x++)
    for(int y=0;y<n;y++)
        H_IP[t*n+u]+=GAMMA[((t*n+y)*n+v)*n+x]*0.5*
                      (g2[((u*n+y)*n+v)*n+x]);//restricted variant


    //H_EA
    set_zero_matr(H_EA,n*n);

    for(int t=0;t<n;t++)
    for(int u=0;u<n;u++)
    for(int v=0;v<n;v++)
        H_EA[t*n+u]+= U_EA[t*n+v]*g1[u*n+v];//restricted variant

    for(int t=0;t<n;t++)
    for(int u=0;u<n;u++)
    for(int v=0;v<n;v++)
    for(int x=0;x<n;x++){
        H_EA[t*n+u]+=U_IP[x*n+v]*(2*g2[((t*n+u)*n+v)*n+x]-g2[((t*n+x)*n+v)*n+u]);
    }

    for(int t=0;t<n;t++)
    for(int u=0;u<n;u++)
    for(int v=0;v<n;v++)
    for(int w=0;w<n;w++)
    for(int x=0;x<n;x++)
        H_EA[t*n+u]-=GAMMA[((v*n+t)*n+x)*n+w]*0.5*
                      (      g2[((v*n+u)*n+w)*n+x]);//restricted variant

    return;
}

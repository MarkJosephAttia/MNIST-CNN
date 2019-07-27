#include "layer.h"

layer:: layer(int myin, int myout, double* myalfa,int*  mynPat, double* drops)
    :nIn(myin), nOut(myout),pAlfa(myalfa),pnPat(mynPat), DropOut1000(drops)
    {
    int i, j;
    w     = matD(nOut,nIn);
    dw    = matD(nOut,nIn);
    b     = matD(nOut);
    db    = matD(nOut);
    mOutF = matD(nOut);
    mOutB = matD(nIn);
    Drop = new int[nOut];
    for(i=0;i<nOut;i++)
     {
      for(j=0;j<nIn;j++)  w[i][j]=((rand()%100)-50)/200.00;
      b[i]=((rand()%100)-50)/200.00;
     }
    }
////////////////////////////////////////////////////////////////////
void layer:: Reset()
{
   int i,j;
   for(i=0;i<nIn; i++) for(j=0;j<nOut;j++) dw[j][i]=0;
   for(j=0;j<nOut;j++)
   {
       db[j]=0;
       if(rand()%1000 < *DropOut1000) Drop[j] = 1;
       else Drop[j]=0;
   }
}
////////////////////////////////////////////////////////////////////
void layer::BP () // to update dl/dw   (can improve)
  {
   int i,j;
   double dz;
   for(i=0;i<nIn; i++) mOutB[i]=0;
   for(j=0;j<nOut;j++)
    {
        if(Drop[j] == 1) continue;
     dz=pInB[j]*(1.2-abs(mOutF[j]));
     db[j]+= dz;
     for(i=0;i<nIn;i++) {
         dw[j][i]+= dz*pInF[i];
         mOutB[i]+=w[j][i]*dz;
         }
    }

  }
////////////////////////////////////////////////////////////////////
  void layer::FF() // to get layer output no mask just batch
  {
  int i,j;
  double z,a,mul;
  mul = 1.0 + (*DropOut1000/1000.0);
  for(j=0;j<nOut;j++){
        if(Drop[j] == 1) {mOutF[j]=0;continue;}
       z=b[j];
       for(i=0;i<nIn;i++) z+=w[j][i]*pInF[i];
       //if (z>1)a=1; else if(z<-1)a=-1; else a=z;
       mOutF[j]=tanh(z*mul);
       }
   }

 //////////////////////////////////////////////////////////////////////
 void layer::Updat()  // to update w and b
 {
    int i,j,num;
    num = *pbatIC+1;
    for(j=0;j<nOut;j++)
     {
       b[j]+=db[j]*(*pAlfa/num);
       for(i=0;i<nIn;i++) w[j][i]+=dw[j][i]*(*pAlfa/num);
      }
  }
//////////////////////////////////////////////////////////////////////
void layer::makeBefore(layer* L) //connect phantom before L
{
     pInB=L->mOutB;
     L->pInF= mOutF;
}
//////////////////////////////////////////////////////////////////////
void layer::print()
{
    int i,j;
    //int nPat=*tr->pnPat;
    cout << "No of Input    ="<< nIn  <<  endl;
    cout << "No of Output   ="<< nOut <<  endl;
    cout << "No of Patterns ="<< (*pnPat) <<  endl;
    cout << "Alfa Value     ="<< (*pAlfa) <<  endl;

    for(i=0;i<nOut  ;i++)
    {
        cout<<"w["<<i+1<<"] = ";
        for(j=0;j<nIn;j++) cout<<w[i][j]<<" , ";
        cout<< "b["<<i+1<<"] = "<<b[i]<<endl;
    }
}
////////////////////////////////////////////////////////////////////////

#include "trainer.h"
int cy = 50;

trainer::trainer(net* theNet,trainSet* myts): ts(myts), Net(theNet)
{
    pnIn   = &(ts->nIn);
    pnOut  = &(ts->nOut);
    pnPat  = &(ts->nPat);
    py = ts->y;
    px = ts->x;
    mda  = matD(*pnOut);     // dl/da
    batI = matI(*pnPat);     // batch index for dynamic pattern selection
    mBatchCount=0;           // to count number of mini-batch done.
    initBatI(100);
    mode= minBat;
    ptrnI=0;                 // pattern pointer

    // to connect trainer to the network and training set.
    Net->Ls[Net->nL-1]->pInB  = mda;      // connect input for last layer for BP
    pa = Net->Ls[Net->nL-1]->mOutF;       // connect the output of the last layer to trainer

    // to adjust the layer data
    for ( int i=0; i<Net->nL; i++){
        Net->Ls[i]->pbatIC = &batIC;          // pointer batch index count
    }
}
////////////////////////////////////////////////////////////////////////
  int trainer::delta()
 {
   int j;
   int nError=0;      // number of outputs with error
   double error=0;

   for(j=0; j<*pnOut; j++) // for all outputs
   {
      mda[j] = (py[ptrnI][j]-pa[j]);
      //cout<<pa[j]<<"  ";
      error = abs(mda[j]);
      Loss += error;
      if (error>MaxError) MaxError=error;
      nError += error>0.6;
   }
   if (nError>0)    //  pattern still has error
     {
       patternInEror++;
     }
   return nError;
 }

  ////////////////////////////////////////////////////////////////
  void   trainer:: train (int cycles)
  {
   int i,j,k;
   int nError; patternInEror=0;     // number of outputs with error

   for ( j=0; j<cycles; j++)
     {
      if(j%2 == 1 && patternInEror == 0) {initBatI(1); continue;}
      MaxError=0; Loss=0; patternInEror=0;   // to collect statistics.
      i=0;
      NReset();
      while(i<batIC)    //for each location in the batch index table
      {
         k=batI[i];    // actual pattern number in the training set
         ptrnI=k;      // the current pattern to be trained
         //if(kbhit()) cy=getch();
         NFF ();
         nError=delta();
         NBP ();
         i++;
      }
      //cout<<patternInEror<<", ";
      if(patternInEror != 0) NUpdat();
      if(j%1000 == 0 && cycles != 1) cout<<j/1000;

      if (j%2==1){
         mBatchCount++;
         //cout<<"-->"<< j<<endl;
         initBatI(1);     // to fill table
      }

   }
  }
  ////////////////////////////////////////////////////////////////
  void trainer::NReset () // to reset dw and db for the whole network
  {
     int i;
     int nlayer;        // number of layers in the net
     nlayer= Net->nL;
     for (i=0; i<nlayer; i++ ) (*Net)[i]->Reset();
  }

  ////////////////////////////////////////////////////////////////
  void  trainer:: NFF ()   // work with one pattern
  {
    int i;
    int nlayer;        // number of layers in the net
    nlayer= Net->nL;
    (*Net)[0]->pInF= px[ptrnI];   // to point t the pattern selected
    for (i=0; i<nlayer; i++ ) (*Net)[i]->FF();
  }
  ////////////////////////////////////////////////////////////////
  void  trainer:: NBP ()
  {
    int i;
    int nlayer;        // number of layers in the net
    nlayer= Net->nL;
    for (i=nlayer-1; i>=0; i-- )  (*Net)[i]->BP();
  }
////////////////////////////////////////////////////////////////
  void  trainer:: NUpdat ()
  {
    int i;
    int nlayer;        // number of layers in the net
    nlayer= Net->nL;
    for (i=nlayer-1; i>=0; i-- )(*Net)[i]->Updat();

  }

////////////////////////////////////////////////////////////////
  void   trainer::initBatI()
  {
      int i;
      for (i=0; i<*pnPat; i++)   batI[i]=i;
      batIC=*pnPat;
  }
////////////////////////////////////////////////////////////////
// to partially fell the index table
  void   trainer::initBatI(int fellPrecent)
  {
      int i,j;
      for (i=0,j=0; i<*pnPat; i++)
      {
         if ((rand()%100)<fellPrecent) // include in the training set
         {
            batI[j]=i;
            j++;
         }
      }
      batIC=j;
  }
////////////////////////////////////////////////////////////////
void  trainer::printTs_out()
 {
    int i,j;
    cout<<"Ts="<<endl;
    for(i=0;i<*pnPat;i++)
    {
     ptrnI=i;   // the current pattern to be trained
     NFF ();
     for(j=0;j<*pnIn;j++)
         cout<<setw(3)<<px[i][j]<<" ";
     cout<<"=";
     for(j=0;j<*pnOut;j++) cout<<setw(3)<<py[i][j];
     for(j=0;j<*pnOut;j++) cout<<setw(3)<<pa[j]<<" E=";
     for(j=0;j<*pnOut;j++) cout<<setw(3)<<abs(pa[j]-py[i][j]);
     cout<<endl;
    }
 }
 ////////////////////////////////////////////////////////////////

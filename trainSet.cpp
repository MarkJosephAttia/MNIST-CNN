#include "trainSet.h"

trainSet::trainSet()
{
}
////////////////////////////////////////////////////////////////////////
void trainSet::Creat()
{

    x = matD(nPat,nIn);       // full input data
    y = matD(nPat,nOut);      // full output data
}
////////////////////////////////////////////////////////////////////////
 void trainSet::XfillRand( int p) // to fill x with random variable
 {
 for(int j=0;j<nIn;j++)
   {
     for(int n=0;n<nPat;n++)
     {
     double r = rand() % 100;
     if(r>p) r=1;
     else    r=-1;
     x[n][j]=r;
     }
   }
 }
  ////////////////////////////////////////////////////////////////////////
  void  trainSet:: XfillBin () // to fill x with binary value
  {
     int i, j,a;
     for ( i=0; i< nPat; i++)
     {
        a=i;
      for (j=0; j<nIn; j++)
      {
         if ((a%2)==0) x[i][j]=1; else x[i][j]=-1;
         a=a/2;

      }
     }
  }
   ////////////////////////////////////////////////////////////////////////
 void trainSet::YfillParity () // to fill y with parity.
 {
    int i,j,pluss;
    for(j=0;j<nPat;j++)
    {
        pluss=0;
        for(i=0;i<nIn;i++)
            if(x[j][i]==1)   pluss+=1;
        y[j][0]=-1;
        for(i=1;i<=nIn;i+=2)
        if  (pluss==i)  y[j][0]=1;

    }
 }
////////////////////////////////////////////////////////////////
void  trainSet::printTs()
 {
    int i,j;
    cout<<"Ts="<<endl;
    for(i=0;i<nPat;i++)
    {
        for(j=0;j<nIn;j++)
            cout<<setw(3)<<x[i][j]<<" ";
        cout<<"="<<setw(9);
        for(j=0;j<nOut;j++)   cout<<setw(3)<<y[i][j];
        cout<<endl;
    }
 }

 ////////////////////////////////////////////////////////////////
void  trainSet::fillsmallXor () //wrong
{
    x[0][0]=-1; x[0][1]=-1; x[0][2]=1;  x[0][3]=1;
    x[1][0]=-1; x[1][1]=1;  x[1][2]=-1; x[1][3]=1;
    y[0][0]=-1; y[0][1]=1;  y[0][2]=1;  y[0][3]=-1;
}
/////////////////////////////// read integer from mnist//////////////////
 int trainSet::readI(ifstream* inDataFile)
{
   unsigned char a,b,c,d;
   unsigned int r=0;
   inDataFile->read((char*)(&a), sizeof(char));
   inDataFile->read((char*)(&b), sizeof(char));
   inDataFile->read((char*)(&c), sizeof(char));
   inDataFile->read((char*)(&d), sizeof(char));
   r=d+256*c+65536*b+16777216*a;
   return (int)r;
}
/////////////////////////////readIm  mnist/////////////////////////////////////
void trainSet::readIm(char* name)
{
   int mn, r,w;
   unsigned char* p;
   ifstream imF;
   imF.open("TRIMG",ios::binary|ios::in);
	if (!imF) {
		cout << "Unable to open file TRIMG";
		exit(1);   // call system to stop
	}
   mn=readI(&imF); nPat=readI(&imF); r=readI(&imF);  w=readI(&imF);
   nIn=r*w;
   p =(unsigned char *) malloc(nIn*nPat);
   imF.read((char*)(p), nIn*nPat);
   imF.close();
   //cout<<mn<<"  "<<nPat<<"  "<<r<<"  "<<w<<endl;
   //====================all data flat but double======================
   double *  pat= (double *) malloc(sizeof(double)*nIn*nPat);
   int i;
   for (i=0; i<nIn*nPat; i++) pat[i]=(p[i]-127.5)/127.5; // from -1 to 1
   //==================== as a two dimensional array====================
   x=(double**) malloc(sizeof(double*)*nPat);
   for (i=0; i<nPat; i++) x[i]= pat+i*nIn;
   delete [] p;
   //never do "delete [] pat;"   it is the only place with pattern data
}
////////////////////////////////readLABEL/////////////////////////////////
void trainSet::readLABEL(char* name, int num )
{
   int mn,i,j;
   unsigned char* imL;
   ifstream labF;
   labF.open("LABEL",ios::binary|ios::in);
	if (!labF) {
		cout << "Unable to open file LABEL";
		exit(1);   // call system to stop
	}
   mn=readI(&labF); nPat=readI(&labF);
   imL= (unsigned char*) malloc(nPat);
   labF.read((char*)(imL), sizeof(char)*nPat);
   labF.close();
   //cout<<mn<<"  "<<nPat<<endl;
   if( num==10){
      //====================all data flat but double 0-9 ====================
      double *  label= (double *) malloc(sizeof(double)*10*nPat);
      for (i=0; i<nPat; i++)
         for (j=0; j<10; j++){
            if (j==imL[i]) label[10*i+j]=1.0;
            else label[10*i+j]=-1.0;
         }

      //As a two dimensional array ====
      nOut=10;   // transformed from an integer from 0 to 9
      y=(double**) malloc(sizeof(double*)*nPat);
      for (i=0; i<nPat; i++) y[i]= label+i*10;
      }   // end of 10
   else
      {
      //==================all data flat but double single num ==========
      double *  label= (double *) malloc(sizeof(double)*nPat);
      for (i=0; i<nPat; i++){
            if (imL[i]==num) label[i]=1.0;
            else label[i]=-1.0;
         }

     //As a two dimensional array ====
      nOut=1;   // transformed from to single
      y=(double**) malloc(sizeof(double*)*nPat);
      for (i=0; i<nPat; i++) y[i]= label+i;
      }

   delete []imL;
   //never do "delete [] label;"   it is the only place with pattern data
}
///////////////////////////////display image from mnist//////////////////
 void trainSet::displayIm(int pI )
{
   int i,j;
    for( i=0; i<28; i++){
         for( j=0; j<28; j++){
               if (i==0&&j==0)   {cout<<'.'; continue;}
               if (i==27&&j==27) {cout<<'.'; continue;}
            if(x[pI][j+28*i]<-.6 ) cout<<char(32)<<char(32);
            else if(x[pI][28*i+j]<-.2) cout<<char(176)<<char(176);
            else if(x[pI][28*i+j]<.2) cout<<char(177)<<char(177);
            else if(x[pI][28*i+j]<.56) cout<<char(178)<<char(178);
            else cout<<char(219)<<char(219);
         }
         cout<<endl;
      }

      for( i=0; i<10; i++){
            if(y[pI][i]<=0 ) cout<<char(176)<<char(176);
            else cout<<char(219)<<char(219);
      }
      cout<<endl;
      for( i=0; i<10; i++){
            if(y[pI][i]<=0 ) cout<<"  ";
            else cout<<i<<' ';
      }
      cout<<endl<<"=================================================================="<<endl;
}
///////////////////////////////to load mnist//////////////////
    void trainSet::loadMnist(int num)
    {
      char IMFname[]="TRIMG";
      char LFname []="LABEL";
      readIm   (IMFname);
      readLABEL(LFname,num); // num =10 means all 10 outputs
    }
///////////////////////////////to load mnist//////////////////
    void trainSet::loadMnistNum(int num,trainSet* t)
    {
        x = t->x;
        nIn = t->nIn;
        nOut = t->nOut;
        nPat = t->nPat;
        char LFname []="LABEL";
        readLABEL(LFname,num); // num =10 means all 10 outputs
    }
///////////////////////////////
unsigned char* trainSet::ReadTest()
{
    int mn, r,w;
   unsigned char* p;
   ifstream imF;
   imF.open("TSTIMG",ios::binary|ios::in);
	if (!imF) {
		cout << "Unable to open file TSTIMG";
		exit(1);   // call system to stop
	}
   mn=readI(&imF); nPat=readI(&imF); r=readI(&imF);  w=readI(&imF);
   nIn=r*w;
   p =(unsigned char *) malloc(nIn*nPat);
   imF.read((char*)(p), nIn*nPat);
   imF.close();
   int i,j;
   for (i=0; i<nPat; i++)
   {
       for (j=0; j<nIn; j++)
       {
           x[i][j] = (p[i*nIn+j]-127.5)/127.5; // from -1 to 1
       }
   }
   //==================== as a two dimensional array====================
   delete [] p;
   /////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////////
   unsigned char* imL;
   ifstream labF;
   labF.open("TSTLABEL",ios::binary|ios::in);
	if (!labF) {
		cout << "Unable to open file LABEL";
		exit(1);   // call system to stop
	}
   mn=readI(&labF); nPat=readI(&labF);
   imL= (unsigned char*) malloc(nPat);
   labF.read((char*)(imL), sizeof(char)*nPat);
   labF.close();
   return imL;
}

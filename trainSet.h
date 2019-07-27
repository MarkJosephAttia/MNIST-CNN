#ifndef TRAINSET_H
#define TRAINSET_H
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <ctime>
#include "matrix.h"
#include "layer.h"
#include<fstream>
using namespace std;
class trainSet
{
    public:
        int nIn;            // input,
        int nOut;           // output,
        int nPat;           // patterns
        double ** x;        // input
        double ** y;        // output

        trainSet();
        void Creat(); // to actually construct the TS.
        trainSet(int mynIn, int mynOut, int mynPat); // input, output, patterns

        void   printTs();
        void   XfillRand( int v);   // to fill x with random variable
        void   XfillBin ();         // to fill x with binary value
        void   YfillParity ();      // to fill y with parity.
        void   fillsmallXor ();     // 2 input xor.
        // for mnist
        int readI(ifstream* inDataFile); // to read integer from mnist file
        void readIm(char* name);
        void readLABEL(char* name,int num);
        void displayIm(int pI);
        void loadMnist(int num =10);
        void loadMnistNum(int num,trainSet* t);
        unsigned char* ReadTest();
    private:
};

#endif // TRAINSET_H

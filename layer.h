#ifndef LAYER_H
#define LAYER_H
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <ctime>
#include "matrix.h"
#include "trainSet.h"
#include "trainer.h"

using namespace std;
class trainSet;
class trainer;
class layer
{
    public:
    int nIn;
    int nOut;
    double* pAlfa;
    double** w ;
    double** dw;
    double*  b;
    double*  db;
    double* mOutF;  //output a
    double* mOutB;  // dl/da1 from second layer and dl/da from training set.
    double* pInF;   // input a
    double* pInB;   // dl/da from output or dl/da1 from second to first (delta roul)
    trainer * tr;    // trainer pointer
    int*   pnPat;
    int*   pbatIC;   // pointer batch index count
    double* DropOut1000;
    int* Drop;

    layer(int myin, int myout, double* myalfa,int*  mynPat, double* drops);
    void makeBefore(layer* L); //connect phantom before L
    void print();
    void printOut();
    void Reset();  // reset dw and db
    void FF();    // to  FF
    void BP();    // to BP
    void Updat(); // layer update
    private:
};

#endif // LAYER_H

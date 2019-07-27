#ifndef NET_H
#define NET_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <ctime>
#include "matrix.h"
#include "layer.h"
#include "trainSet.h"
using namespace std;
class trainSet;
class trainer;
class layer;
class net
{
    // we assume the input is not a layer but output is
    public:
    int  nL;           // number of layers
    int  nC;
    int* pnIn;        // pointer to number of inputs.
    int* FiltersNumber;
    int* FiltersLength;
    int* nForLayers;   // size of all layers
    int* pnPat;    // num of patterns in the training set (j)
    double alfa;
    trainSet* ts;
    layer** Ls;
    double DropOut1000;

    net(int mynL,trainSet* myts);
    layer* operator [] (int i); // to return a pointer to layer.
    void Creat();
    void print();
};


#endif // NET_H

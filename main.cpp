#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <ctime>
#include "matrix.h"
#include "trainSet.h"
#include "layer.h"
#include "net.h"
#include "trainer.h"
#include "ConvolutionN.h"
#include <thread>
#include <unistd.h>

using namespace std;

void Train(trainer* trFC)
{
    trFC->train(40000);                       //Change it to 40,000 for real training
    for(int s = 0; s < 60; s++)              //Change it to 60 for real training
    {
        //if(s==35) trFC->Net->alfa = 0.5;
        if(s==30) trFC->Net->alfa = 0.5;
        trFC->initBatI();
        trFC->train(1);
        if(trFC->patternInEror == 0) break;
    }
}

int main()
{
    srand(time(NULL));
    int* Errors = new int[10];
    int i;
    int k = 0;
    unsigned char* numberIs;
    trainSet** TS = new trainSet*[10];
    TS[0] = new trainSet();
    TS[0]->loadMnist(0);    // initialize x,y, nIn, nOut, nPat
    //while (k<TS->nPat){TS->displayIm(k); k++; getche(); }

    ConvN* C = new ConvN(TS[0],2,Avg);
    C->nFlayer[0]=6;
    C->nFlayer[1]=16;
    C->Fsize=5;
    C->PaddStep=0;
    C->MainD=28;
    C->Creat();
    C->train(60000);
    double** hh = TS[0]->x;
    TS[0]->x = C->FinalOut2;
    TS[0]->nIn = 256;
    net** NFC = new net*[10];
    trainer**  trFC = new trainer*[10];
    thread th[10];
    for(int num = 0; num < 10; num++)
    {
        if(num != 0) TS[num] = new trainSet();
        TS[num]->loadMnistNum(num,TS[0]);
        NFC[num] = new net(3,TS[num]);  //3
        NFC[num]->nForLayers[0]= 50;    //50
        NFC[num]->nForLayers[1]= 50;    //50
        //NFC[num]->nForLayers[2]= 35;
        NFC[num]->Creat();
        NFC[num]->DropOut1000 = 100.0;
        trFC[num] = new trainer(NFC[num], TS[num]);
        NFC[num]->alfa = 0.7;
        th[num] = thread(Train,trFC[num]);
        cout<<"Thread "<<num<<" Started"<<endl;
    }
    for(int endd = 0; endd < 10; endd++)
    {
        th[endd].join();
        NFC[endd]->DropOut1000 = 0.0;
        trFC[endd]->NReset();
    }
    for(int endd = 0; endd < 10; endd++)
    {
        Errors[endd] = 0;
        cout<<endl<<"Number ("<<endd<<") ErrorPatterns = " <<trFC[endd]->patternInEror<<" And Loss = "<<trFC[endd]->Loss<<endl;
    }
    ///////////////////////////////TestCode////////////////////////////////
    TS[0]->x = hh;
    numberIs = TS[0]->ReadTest();
    C->train(10000);
    TS[0]->x = C->FinalOut2;
    TS[0]->nIn = 256;
    double maximum = -1.1; int index = -1;
    int Count = 0;
    for(int pat = 0; pat < 10000; pat++)
    {
        maximum = -1.1; index = -1;
        for(int num = 0; num < 10; num++)
        {
            TS[num]->nPat = 10000;
            trFC[num]->ptrnI = pat;
            trFC[num]->NFF();
            if(trFC[num]->pa[0] > maximum)
            {
                maximum = trFC[num]->pa[0];
                index = num;
            }
        }
        if(index != (int)numberIs[pat])
        {
            Count++;
            Errors[(int)numberIs[pat]]++;
        }
        //if(pat > 9980) {print(hh[pat],28,28); cout<<endl<<index<<endl;}
        //print(trFC[0]->px[pat],28,28); cout<<endl<<index<<endl;
    }

    cout<<endl<<"             //////TestSet//////"<<endl;
    for(int endd = 0; endd < 10; endd++)
    {
        cout<<endl<<"Number <"<<endd<<"> ErrorPatterns = " <<Errors[endd]<<endl;
    }
    cout<<endl<<"              Test Error is: "<<Count<<endl;

    //tr->printTs_out();
    //cout<<endl<<i<<", "<<trFC->Loss<<", "<< trFC->patternInEror<<endl;// getch();

    return 0;
}


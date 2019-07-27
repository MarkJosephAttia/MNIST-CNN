#include "net.h"


////////////////////////////////////////////////////////////////////
net::net(int mynL,trainSet* myts): nL(mynL), ts(myts)
{
    // we assume the input is not a layer but output is
    Ls = new layer*[nL];
    nForLayers= new int[nL];
    pnIn= &(ts->nIn);              // number of inputs per pattern
    pnPat=&(ts->nPat);            // number of patterns in the training set
    DropOut1000 = 0;
    nForLayers[nL-1]=ts->nOut;     // number of outputs per pattern
}

////////////////////////////////////////////////////////////////////
void net::Creat()
{
    Ls[0]= new layer(*pnIn,nForLayers[0],&alfa,pnPat,&DropOut1000); // input to first layer

    double* zer = new double[1];
    zer[0] = 0.0;
    for (int i=1; i<nL; i++)
    {
        if(i != nL-1)Ls[i]= new layer(nForLayers[i-1],nForLayers[i],&alfa,pnPat,&DropOut1000);
        else Ls[i]= new layer(nForLayers[i-1],nForLayers[i],&alfa,pnPat,zer);
        Ls[i-1]->makeBefore(Ls[i]);
    }
}


//////////////////////////////////////////////////////////////
layer* net::operator [] (int i) // to return a pointer to layer.
{
    return Ls[i];
}
//////////////////////////////////////////////////////////////
void net::print()
{
    int i;
    cout<<endl<< "network information"<<endl;
    cout<<"number of layers= "<<nL<< endl;
    cout<<"number of inputs"<< *pnIn<<endl;
     for (i=0; i<nL; i++){
        cout<<"number of neurons in layer "<<i<<"= " <<nForLayers[i]<<" location= "<< Ls[i]<<endl;
     }

    for (i=0; i<nL; i++)
    {
        cout<<endl<<i;
        Ls[i]->print();
    }

}

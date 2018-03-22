#include <math.h>
#include "cec17_test_func.cpp"
#include <iostream>
#include <time.h>
#include <fstream>
#include <random>

using namespace std;
unsigned globalseed = unsigned(time(NULL));
//unsigned globalseed = 2018;
unsigned seed1 = globalseed+0;
unsigned seed2 = globalseed+100;
unsigned seed3 = globalseed+200;
unsigned seed4 = globalseed+300;
unsigned seed5 = globalseed+400;
std::mt19937 generator_uni_i(seed1);
std::mt19937 generator_uni_r(seed2);
std::mt19937 generator_norm(seed3);
std::mt19937 generator_cachy(seed4);
std::mt19937 generator_uni_i_2(seed5);
std::uniform_int_distribution<int> uni_int(0,32768);
std::uniform_real_distribution<double> uni_real(0.0,1.0);
std::normal_distribution<double> norm_dist(0.0,1.0);
std::cauchy_distribution<double> cachy_dist(0.0,1.0);

int IntRandom(int target)
{
    if(target == 0)
        return 0;
    return uni_int(generator_uni_i)%target;
}
double Random(double minimal, double maximal)
{
    return uni_real(generator_uni_r)*(maximal-minimal)+minimal;
}
double NormRand(double mu, double sigma)
{
    return norm_dist(generator_norm)*sigma + mu;
}
double CachyRand(double mu, double sigma)
{
    return cachy_dist(generator_cachy)*sigma+mu;
}
void qSort2int(double* Mass,int* Mass2, int low, int high)
{
    int i=low;
    int j=high;
    double x=Mass[(low+high)>>1];
    do
    {
        while(Mass[i]<x)    ++i;
        while(Mass[j]>x)    --j;
        if(i<=j)
        {
            double temp=Mass[i];
            Mass[i]=Mass[j];
            Mass[j]=temp;
            double temp2=Mass2[i];
            Mass2[i]=Mass2[j];
            Mass2[j]=temp2;
            i++;    j--;
        }
    } while(i<=j);
    if(low<j)   qSort2int(Mass,Mass2,low,j);
    if(i<high)  qSort2int(Mass,Mass2,i,high);
}

void cec17_test_func(double *, double *,int,int,int);
double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;
int GNVars;
double stepsFEval [] = {0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
double ResultsArray[51][14];
int LastFEcount;
int NFEval = 0;
int MaxFEval = 0;
double tempF[1];
double fopt[1];
char buffer[30];
double globalbest;
bool globalbestinit;
bool initfinished;
vector<double> FitTemp;

void GetOptimum(int func_num, double* xopt, double* fopt)
{
    FILE *fpt;
    char FileName[30];
    sprintf(FileName, "input_data/shift_data_%d.txt", func_num);
    fpt = fopen(FileName,"r");
    if (fpt==NULL)
        printf("\n Error: Cannot open input file for reading \n");
    for(int k=0;k<GNVars;k++)
        fscanf(fpt,"%lf",&xopt[k]);
    fclose(fpt);
    cec17_test_func(xopt, fopt, GNVars, 1, func_num);
}
void SaveBestValues(int RunN, double newbestfit)
{
    for(int stepFEcount=LastFEcount;stepFEcount<14;stepFEcount++)
    {
        if(NFEval == int(stepsFEval[stepFEcount]*MaxFEval))
        {
            double temp = globalbest - fopt[0];
            if(temp <= 10E-8)
                temp = 0;
            ResultsArray[RunN][stepFEcount] = temp;
            LastFEcount = stepFEcount;
        }
    }
}
void FindLimits(double* Ind, double* Parent,int CurNVars,double CurLeft, double CurRight)
{
	for (int j = 0; j<CurNVars; j++)
	{
		if (Ind[j] < CurLeft)
            Ind[j] = (CurLeft + Parent[j])/2.0;
		if (Ind[j] > CurRight)
            Ind[j] = (CurRight + Parent[j])/2.0;
	}
}
class Optimizer
{
public:
    bool FitNotCalculated;
    /*параметры*/
    double F;
    double F2;
    double Cr;
    int Int_ArchiveSizeParam;
    int MemorySize;
    int MemoryIter;
    int SuccessFilled;
    int MemoryCurrentIndex;

    int NVars;			    // размерность пространства
    int NInds;			    // число частиц
    int NIndsMax;
    int NIndsMin;

    double bestfit;
    int besti;

    int func_num;
    int TheChosenOne;
    int Generation;
    double ArchiveSizeParam;
    double psize;
    double psizeParam;
    int ArchiveSize;
    int CurrentArchiveSize;
    double ArchiveProb;

    double* Donor;
    double* Trial;
    int* Rands;

    double** Popul;	        // массив для частиц
    double** PopulTemp;
    double* FitMass;		// значения функции пригодности
    double* FitMassTemp;
    double* FitMassCopy;
    double* BestInd;	    // лучшее найденное решение
    int* Indexes;
    double** Archive;
    double* FitMassArch;

    double* tempSuccessCr;
    double* tempSuccessF;

    double Right;		    // верхняя граница
    double Left;		    // нижняя граница

    double* MemoryCr;
    double* MemoryF;
    double* FitDelta;

    void Initialize(int newNInds, int newNVars, int func_num,
                    double NewArchSizeParam, double NewArchiveProbParam, double NewPSize);
    void Clean();
    void MainCycle(int RunN);
    void FindNSaveBest(bool init);

    void PSO_MoveP();
    void PSO_UpdateBests(int TheChosenOne);

    void CopyToArchive(double* RefusedParent,double RefusedFitness);
    void SaveSuccessCrF(double Cr, double F, double FitD);
    void UpdateMemoryCrF();
    double MeanWL(double* Vector, double* TempWeights, int Size);
    void RemoveWorst(int NInds, int NewNInds);

    int GetImprState();
    bool GetFitState();

    int SelectWorst();
    int SelectBest();
};
double cec_17_(double* HostVector,int func_num)
{
    cec17_test_func(HostVector, tempF, GNVars, 1, func_num);
    NFEval++;
    return tempF[0];
}
void Optimizer::Initialize(int newNInds, int newNVars, int newfunc_num,
                          double NewArchSizeParam, double NewArchiveProbParam, double NewPSize)
{
    FitNotCalculated = true;
    NInds = newNInds;
    NIndsMax = NInds;
    NIndsMin = 4;
    NVars = newNVars;
    Left = -100;
    Right = 100;
    Cr = 0.5;
    F = 0.8;
    besti = 0;
    Generation = 0;
    TheChosenOne = 0;
    CurrentArchiveSize = 0;
    psizeParam = NewPSize;
    ArchiveSizeParam = NewArchSizeParam;
    Int_ArchiveSizeParam = ceil(ArchiveSizeParam);
    ArchiveSize = NIndsMax*ArchiveSizeParam;
    ArchiveProb = NewArchiveProbParam;
    func_num = newfunc_num;

    Popul = new double*[NIndsMax];
    for(int i=0;i!=NIndsMax;i++)
        Popul[i] = new double[NVars];
    PopulTemp = new double*[NIndsMax];
    for(int i=0;i!=NIndsMax;i++)
        PopulTemp[i] = new double[NVars];
    Archive = new double*[NIndsMax*Int_ArchiveSizeParam];
    for(int i=0;i!=NIndsMax*Int_ArchiveSizeParam;i++)
        Archive[i] = new double[NVars];
    FitMassArch = new double[NIndsMax*Int_ArchiveSizeParam];
    FitMass = new double[NIndsMax];
    FitMassTemp = new double[NIndsMax];
    FitMassCopy = new double[NIndsMax];
    Indexes = new int[NIndsMax];
    BestInd = new double[NVars];

	for (int i = 0; i<NIndsMax; i++)
		for (int j = 0; j<NVars; j++)
			Popul[i][j] = Random(Left,Right);

    Donor = new double[NVars];
    Trial = new double[NVars];
    Rands = new int[NIndsMax];

    tempSuccessCr = new double[NIndsMax];
    tempSuccessF = new double[NIndsMax];
    FitDelta = new double[NIndsMax];

    for(int i=0;i!=NIndsMax;i++)
    {
        tempSuccessCr[i] = 0;
        tempSuccessF[i] = 0;
    }

    MemorySize = 5;
    MemoryIter = 0;
    SuccessFilled = 0;

    MemoryCr = new double[MemorySize];
    MemoryF = new double[MemorySize];
    for(int i=0;i!=MemorySize;i++)
    {
        MemoryCr[i] = 0.8+0.0*Random(0,1);
        MemoryF[i] = 0.3+0.0*Random(0,1);
    }
}
void Optimizer::SaveSuccessCrF(double Cr, double F, double FitD)
{
    tempSuccessCr[SuccessFilled] = Cr;
    tempSuccessF[SuccessFilled] = F;
    FitDelta[SuccessFilled] = FitD;
    SuccessFilled ++ ;
}
void Optimizer::UpdateMemoryCrF()
{
    if(SuccessFilled != 0)
    {
        double Old_F = MemoryF[MemoryIter];
        double Old_Cr = MemoryCr[MemoryIter];
        double tempmax = tempSuccessCr[0];
        for(int i=0;i!=SuccessFilled;i++)
            if(tempSuccessCr[i] > tempmax)
                tempmax = tempSuccessCr[i];
        if(MemoryCr[MemoryIter] == -1 || tempmax == 0)
            MemoryCr[MemoryIter] = -1;
        else
            MemoryCr[MemoryIter] = (MeanWL(tempSuccessCr,FitDelta,SuccessFilled) + Old_Cr)/2.0;
        MemoryF[MemoryIter] = (MeanWL(tempSuccessF,FitDelta,SuccessFilled) + Old_F)/2.0;
        MemoryIter++;
        if(MemoryIter >= MemorySize)
            MemoryIter = 0;
    }
}
double Optimizer::MeanWL(double* Vector, double* TempWeights, int Size)
{
    double SumWeight = 0;
    double SumSquare = 0;
    double Sum = 0;
    for(int i=0;i!=SuccessFilled;i++)
        SumWeight += TempWeights[i];
    double* Weights = new double[SuccessFilled];

    for(int i=0;i!=SuccessFilled;i++)
        Weights[i] = TempWeights[i]/SumWeight;
    for(int i=0;i!=SuccessFilled;i++)
        SumSquare += Weights[i]*Vector[i]*Vector[i];
    for(int i=0;i!=SuccessFilled;i++)
        Sum += Weights[i]*Vector[i];
    delete Weights;
    if(fabs(Sum) > 0.000001)
        return SumSquare/Sum;
    else
        return 0.5;
}
void Optimizer::CopyToArchive(double* RefusedParent,double RefusedFitness)
{
    if(CurrentArchiveSize < ArchiveSize)
    {
        for(int i=0;i!=NVars;i++)
            Archive[CurrentArchiveSize][i] = RefusedParent[i];
        FitMassArch[CurrentArchiveSize] = RefusedFitness;
        CurrentArchiveSize++;
    }
    else
    {
        int RandomNum = IntRandom(ArchiveSize);
        for(int i=0;i!=NVars;i++)
            Archive[RandomNum][i] = RefusedParent[i];
        FitMassArch[RandomNum] = RefusedFitness;
    }
}
void Optimizer::FindNSaveBest(bool init)
{
    if(FitMass[TheChosenOne] <= bestfit || init)
    {
        bestfit = FitMass[TheChosenOne];
        besti = TheChosenOne;
        for(int j=0;j!=NVars;j++)
            BestInd[j] = Popul[besti][j];
    }
    if(bestfit < globalbest)
        globalbest = bestfit;
}
void Optimizer::RemoveWorst(int NInds, int NewNInds)
{
    int PointsToRemove = NInds - NewNInds;
    for(int L=0;L!=PointsToRemove;L++)
    {
        double WorstFit = FitMass[0];
        int WorstNum = 0;
        for(int i=1;i!=NInds;i++)
        {
            if(FitMass[i] > WorstFit)
            {
                WorstFit = FitMass[i];
                WorstNum = i;
            }
        }
        for(int i=WorstNum;i!=NInds-1;i++)
        {
            for(int j=0;j!=NVars;j++)
                Popul[i][j] = Popul[i+1][j];
            FitMass[i] = FitMass[i+1];
        }
    }
}
void Optimizer::MainCycle(int RunN)
{
    for(int TheChosenOne=0;TheChosenOne!=NInds;TheChosenOne++)
    {
        FitMass[TheChosenOne] = cec_17_(Popul[TheChosenOne],func_num);
        FindNSaveBest(TheChosenOne == 0);
        if(!globalbestinit || bestfit < globalbest)
        {
            globalbest = bestfit;
            globalbestinit = true;
        }
        SaveBestValues(RunN, bestfit);
    }
    do
    {
        double minfit = FitMass[0];
        double maxfit = FitMass[0];
        int prand;
        minfit = FitMass[0];
        maxfit = FitMass[0];
        for(int i=0;i!=NInds;i++)
        {
            FitMassCopy[i] = FitMass[i];
            Indexes[i] = i;
            if(FitMass[i] >= maxfit)
                maxfit = FitMass[i];
            if(FitMass[i] <= minfit)
                minfit = FitMass[i];
        }
        if(minfit != maxfit)
            qSort2int(FitMassCopy,Indexes,0,NInds-1);
        FitTemp.resize(NInds);
        for(int i=0;i!=NInds;i++)
            FitTemp[i] = 3.0*(NInds-i);
        std::discrete_distribution<int> ComponentSelector (FitTemp.begin(),FitTemp.end());

        psize = ((psizeParam/2.0)/(double)MaxFEval*(double)NFEval+psizeParam/2.0);
        int psizeval = NInds*psize;
        if(psizeval <= 1)
            psizeval = 2;

        for(int TheChosenOne=0;TheChosenOne!=NInds;TheChosenOne++)
        {
            MemoryCurrentIndex = IntRandom(MemorySize+1);
            do
                prand = Indexes[IntRandom(psizeval)];
            while(prand == TheChosenOne && (double)NFEval/(double)MaxFEval < 0.5);

            int Rand1;
            int Rand2;
            do
                Rand1 = Indexes[ComponentSelector(generator_uni_i_2)];
            while(Rand1 == prand);
            do
                Rand2 = Indexes[ComponentSelector(generator_uni_i_2)];
            while(Rand2 == prand || Rand2 == Rand1);
            do
            {
                if(MemoryCurrentIndex < MemorySize)
                    F = CachyRand(MemoryF[MemoryCurrentIndex],0.1);
                else
                    F = CachyRand(0.9,0.1);
            }
            while(F < 0.0);
            if(F > 1.0)
                F = 1.0;
            if((double)NFEval/(double)MaxFEval < 0.6 && F > 0.7)
                F = 0.7;
            if((double)NFEval/(double)MaxFEval < 0.2)
                F2 = 0.7*F;
            else if((double)NFEval/(double)MaxFEval < 0.4)
                F2 = 0.8*F;
            else
                F2 = 1.2*F;
            if(Random(0,1) < (double)CurrentArchiveSize/((double)CurrentArchiveSize+(double)NInds))
            {
                Rand2 = IntRandom(CurrentArchiveSize);
                for(int j=0;j!=NVars;j++)
                {
                    Donor[j] = Popul[TheChosenOne][j] +
                        F2*(Popul[prand][j] - Popul[TheChosenOne][j]) +
                        F*(Popul[Rand1][j] - Archive[Rand2][j]);
                }
            }
            else
            {
                for(int j=0;j!=NVars;j++)
                {
                    Donor[j] = Popul[TheChosenOne][j] +
                        F2*(Popul[prand][j] - Popul[TheChosenOne][j]) +
                        F*(Popul[Rand1][j] - Popul[Rand2][j]);
                }
            }
            FindLimits(Donor,Popul[TheChosenOne],NVars,Left,Right);


            int WillCrossover = IntRandom(NVars);
            if(MemoryCurrentIndex < MemorySize)
            {
                if(MemoryCr[MemoryCurrentIndex] < 0)
                    Cr = 0;
                else
                    Cr = NormRand(MemoryCr[MemoryCurrentIndex],0.1);
            }
            else
                Cr = NormRand(0.9,0.1);
            if(Cr >= 1)
                Cr = 1;
            if(Cr <= 0)
                Cr = 0;

            if((double)NFEval / (double)MaxFEval < 0.25)
                Cr = max(Cr,0.7);
            if((double)NFEval / (double)MaxFEval < 0.5)
                Cr = max(Cr,0.6);
            for(int j=0;j!=NVars;j++)
            {
                if(Random(0,1) < Cr || WillCrossover == j)
                    PopulTemp[TheChosenOne][j] = Donor[j];
                else
                    PopulTemp[TheChosenOne][j] = Popul[TheChosenOne][j];
            }
            FitTemp[TheChosenOne] = cec_17_(PopulTemp[TheChosenOne],func_num);
            if(FitTemp[TheChosenOne] <= globalbest)
                globalbest = FitTemp[TheChosenOne];
            if(FitTemp[TheChosenOne] <= FitMass[TheChosenOne])
                SaveSuccessCrF(Cr,F,fabs(FitMass[TheChosenOne]-FitTemp[TheChosenOne]));
            SaveBestValues(RunN,bestfit);
        }

        for(int TheChosenOne=0;TheChosenOne!=NInds;TheChosenOne++)
        {
            if(FitTemp[TheChosenOne] <= FitMass[TheChosenOne])
            {
                CopyToArchive(Popul[TheChosenOne],FitMass[TheChosenOne]);
                for(int j=0;j!=NVars;j++)
                    Popul[TheChosenOne][j] = PopulTemp[TheChosenOne][j];
                FitMass[TheChosenOne] = FitTemp[TheChosenOne];
            }
        }

        int newNInds = int(double(NIndsMin-NIndsMax)/MaxFEval*NFEval + NIndsMax);
        if(newNInds < NIndsMin)
            newNInds = NIndsMin;
        if(newNInds > NIndsMax)
            newNInds = NIndsMax;
        int newArchSize = double(MaxFEval-NFEval)/(double)MaxFEval*(ArchiveSizeParam*(NIndsMax-NIndsMin));
        if(newArchSize < NIndsMin)
            newArchSize = NIndsMin;
        ArchiveSize = newArchSize;
        if(CurrentArchiveSize >= ArchiveSize)
            CurrentArchiveSize = ArchiveSize;
        RemoveWorst(NInds,newNInds);
        NInds = newNInds;
        UpdateMemoryCrF();
        SuccessFilled = 0;
        Generation ++;

    } while(NFEval < MaxFEval);
}
void Optimizer::Clean()
{
    delete Donor;
    delete Trial;
    delete Rands;
    for(int i=0;i!=NIndsMax;i++)
    {
        delete Popul[i];
        delete PopulTemp[i];
    }
    for(int i=0;i!=NIndsMax*Int_ArchiveSizeParam;i++)
        delete Archive[i];
    delete Archive;
    delete Popul;
    delete PopulTemp;
    delete FitMass;
    delete FitMassTemp;
    delete FitMassCopy;
    delete BestInd;
    delete Indexes;
    delete tempSuccessCr;
    delete tempSuccessF;
    delete FitDelta;
    delete MemoryCr;
    delete MemoryF;
}
int main()
{
    cout<<"Random seeds are:"<<endl;
    cout<<seed1<<"\t"<<seed2<<"\t"<<seed3<<"\t"<<seed4<<"\t"<<seed5<<"\n";

    if(false)
    {
        ofstream fout_t("time_complexity.txt");
        cout<<"Running time complexity code"<<endl;
        double T0, T1;
        double x = 0.55;
        unsigned t0=clock(),t1;
        for(int i=1;i!=1000000;i++)
        {
            x = x+x;
            x = x/2.0;
            x = x*x;
            x = sqrt(x);
            x = log(x);
            x = exp(x);
            x = x/(x+2.0);
        }
        t1=clock()-t0;
        T0 = t1;
        cout << "T0 = " << T0 << endl;
        fout_t << "T0 = " << T0 << endl;
        double* xopt;
        for(int i=0;i!=4;i++)
        {
            if(i == 0)
                GNVars = 10;
            if(i == 1)
                GNVars = 30;
            if(i == 2)
                GNVars = 50;
            if(i == 3)
                GNVars =100;
            cout<<"D = "<<GNVars<<endl;
            fout_t<<"D = "<<GNVars<<endl;
            MaxFEval = 200000;
            xopt = new double[GNVars];
            unsigned t0=clock(),t1;
            for(int j=0;j!=MaxFEval;j++)
                cec17_test_func(xopt, tempF, GNVars, 1, 18);
            t1=clock()-t0;
            T1 = t1;
            cout<<"T1 = "<<T1<<endl;
            fout_t<<"T1 = "<<T1<<endl;

            Optimizer OptZ;
            double avgT2 = 0;
            for(int j=0;j!=5;j++)
            {
                unsigned t0=clock(),t1;
                GetOptimum(18,xopt,fopt);
                globalbestinit = false;
                initfinished = false;
                LastFEcount = 0;
                NFEval = 0;
                double NewPopSize = int(75*pow(GNVars,2./3.));
                double NewArchSize = 1.0;
                double NewArchProb = 0.25;
                double NewPsize = 0.17;
                OptZ.Initialize(NewPopSize,GNVars,18,NewArchSize,NewArchProb,NewPsize);
                OptZ.MainCycle(0);
                OptZ.Clean();
                t1=clock()-t0;
                cout<<"T2["<<j<<"] = "<<t1<<endl;
                fout_t<<"T2["<<j<<"] = "<<t1<<endl;
                avgT2 += t1/5.0;
            }
            delete xopt;
            cout<<"Average T2 = " << avgT2 << endl;
            fout_t<<"Average T2 = " << avgT2 << endl;
            cout<<"Algorithm complexity is "<<(avgT2-T1)/T0<<endl;
            fout_t<<"Algorithm complexity is "<<(avgT2-T1)/T0<<endl;
        }
    }

    unsigned t0=clock(),t1;
    double T0, T1;

    for(int GNVarsIter = 0;GNVarsIter!=1;GNVarsIter++)
    {
        if(GNVarsIter == 0)
            GNVars = 10;
        if(GNVarsIter == 1)
            GNVars = 30;
        if(GNVarsIter == 2)
            GNVars = 50;
        if(GNVarsIter == 3)
            GNVars = 100;
        MaxFEval = GNVars*10000;
        cout<<"Run D_"<<GNVars<<"\n";
        Optimizer OptZ;
        double* xopt = new double[GNVars];

        for(int func_num = 1; func_num < 31; func_num++)
        {
            cout<<"Function "<<func_num<<"\t"<<"D"<<GNVars<<"\n"<<"Runs: "<<endl;
            sprintf(buffer, "LSHADE_RSP_%d_%d.txt",func_num,GNVars);
            ofstream fout(buffer);
            for (int RunN = 0;RunN!=51;RunN++)
            {
                cout<<RunN<<"\t";
                GetOptimum(func_num,xopt,fopt);
                globalbestinit = false;
                initfinished = false;
                LastFEcount = 0;
                NFEval = 0;
                double NewPopSize = int(75*pow(GNVars,2./3.));
                double NewArchSize = 1.0;
                double NewArchProb = 0.25;
                double NewPsize = 0.17;
                OptZ.Initialize(NewPopSize,GNVars,func_num,NewArchSize,NewArchProb,NewPsize);
                OptZ.MainCycle(RunN);
                OptZ.Clean();
            }
            cout<<endl;
            for(int i=0;i!=14;i++)
            {
                for(int j=0;j!=51;j++)
                    fout<<ResultsArray[j][i]<<"\t";
                fout<<"\n";
            }
        }
        cout<<endl;
        delete xopt;
    }
    t1=clock()-t0;
    T0 = t1;
    cout << "T0 = " << T0 << endl;
	return 0;
}

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "../include/dense_tile.h"

static void naive_double(const std::vector<float>& A,const std::vector<float>& B,std::vector<float>& C,int M,int N,int K,int lda,int ldb,int ldc){
    for(int i=0;i<M;++i){
        for(int j=0;j<N;++j){
            double acc=0.0;
            for(int k=0;k<K;++k) acc += double(A[i*lda+k])*double(B[k*ldb+j]);
            C[i*ldc+j]=float(acc);
        }
    }
}

static bool approx_equal(const std::vector<float>& R,const std::vector<float>& S){
    size_t n=R.size();
    double max_rel=0.0;
    double max_abs=0.0;
    for(size_t i=0;i<n;++i){
        double a=double(R[i]); double b=double(S[i]);
        double d=fabs(a-b); max_abs = std::max(max_abs,d);
        double rel = (fabs(a) < 1e-4) ? d : d/(fabs(a)+1e-12);
        max_rel = std::max(max_rel, rel);
    }
    std::cout<<"  max_abs="<<max_abs<<" max_rel="<<max_rel<<"\n";
    return (max_rel < 1e-3) && (max_abs < 1e-2);
}

int main(){
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> d(-1.0f,1.0f);

    auto run_case = [&](int M,int N,int K){
        std::vector<float> A(M*K),B(K*N),Cref(M*N),Ctest(M*N);
        for(auto &x:A) x=d(rng);
        for(auto &x:B) x=d(rng);
        std::fill(Cref.begin(),Cref.end(),0.0f);
        std::fill(Ctest.begin(),Ctest.end(),0.0f);
        naive_double(A,B,Cref,M,N,K,K,N,N);
        // call small_gemm_base (in make_llm_high namespace)
        make_llm_high::small_gemm_base(A.data(),B.data(),Ctest.data(),M,N,K,K,N,N);
        bool ok = approx_equal(Cref,Ctest);
        if(!ok){
            std::cout<<"FAIL case M="<<M<<" N="<<N<<" K="<<K<<"\n";
            // print some diffs
            for(int i=0;i<M && i<8;i++){
                for(int j=0;j<N && j<8;j++){
                    size_t idx=i*N+j;
                    std::cout<<" ("<<i<<","<<j<<") ref="<<Cref[idx]<<" test="<<Ctest[idx]<<"\n";
                }
            }
        }
        return ok;
    };

    std::vector<std::tuple<int,int,int>> cases;
    // small sizes and tails
    for(int m=1;m<=16;m++) for(int n=1;n<=16;n++) for(int k=1;k<=16;k+= (k<4?1:4)) cases.emplace_back(m,n,k);
    // some medium sizes including 8x8 and tails
    cases.emplace_back(8,8,512);
    cases.emplace_back(8,8,63);
    cases.emplace_back(8,8,64);
    cases.emplace_back(7,9,128);
    cases.emplace_back(16,8,200);
    cases.emplace_back(32,48,64);
    cases.emplace_back(128,128,64);

    bool all=true;
    for(auto &t:cases){
        int M,N,K; std::tie(M,N,K)=t;
        std::cout<<"Running case M="<<M<<" N="<<N<<" K="<<K<<"\n";
        bool ok = run_case(M,N,K);
        if(!ok){
            std::cout<<"Stopping on first failure.\n";
            return 1;
        }
    }

    if(all) std::cout<<"All micro-kernel tests passed!\n";
    else std::cout<<"Some micro-kernel tests failed.\n";
    return all?0:1;
}

#include "strata/core/spec_hybrid.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
using strata::core::SpecHybrid;
int main(){
    SpecHybrid h(3,128,256);int32_t out[5];
    const std::vector<int32_t> seq{10,20,30,40,50,60,10,20,30,40,50,60,10,20,30};
    h.observe(seq.data(),seq.size());int n=h.propose(5,out);
    assert(n>=2&&out[0]==40&&out[1]==50&&h.last_order()>=2);
    h.reset();assert(h.propose(5,out)==0);
    const std::vector<int32_t> ambiguous{1,2,3,4,1,2,3,5,1,2,3};h.observe(ambiguous.data(),ambiguous.size());assert(h.propose(5,out)==0);
    // Greedy target verification is invariant to draft source and rejects at the first mismatch.
    const int32_t target[]{40,50,99,7,8},draft[]{40,50,60,70,80};int32_t emitted[6];
    int kept=strata::core::greedy_accept_prefix(draft,5,target,emitted,6);
    assert(kept==3&&emitted[0]==40&&emitted[1]==50&&emitted[2]==99);
    // Complete deterministic baseline vs hybrid candidate stream.
    const int32_t model[]{10,20,30,40,50,60,10,20,30,40};std::vector<int32_t> base,hybrid;
    SpecHybrid g(3,128,256);g.observe(seq.data(),seq.size());
    for(int i=0;i<10;++i){base.push_back(model[i]);int k=g.propose(5,out);(void)k;hybrid.push_back(model[i]);g.observe(model[i]);}
    assert(base==hybrid);
    for(int i=0;i<10000;++i){int32_t t=i%17;g.observe(t);g.propose(5,out);}
    // Deterministic target model: autoregressive vs hybrid speculative verification.
    auto next_token=[](const std::vector<int32_t>& h){uint32_t x=2166136261u;for(size_t i=h.size()>8?h.size()-8:0;i<h.size();++i){x^=uint32_t(h[i]);x*=16777619u;}return int32_t(x%97);};
    std::vector<int32_t> prompt{3,1,4,1,5,9,2,6,5,3,5,8,9,7,9};
    std::vector<int32_t> ar=prompt, sp=prompt; SpecHybrid sh(3,256,1024); sh.observe(prompt.data(),prompt.size());
    for(int i=0;i<200;++i) ar.push_back(next_token(ar));
    while(sp.size()<ar.size()){
        int32_t d[5], tgt[5], emitted[6]; int k=sh.propose(5,d); if(k==0){d[0]=-1;k=1;}
        std::vector<int32_t> h=sp; for(int i=0;i<k;++i){tgt[i]=next_token(h);h.push_back(d[i]);}
        int n=strata::core::greedy_accept_prefix(d,k,tgt,emitted,6);
        for(int i=0;i<n&&sp.size()<ar.size();++i){sp.push_back(emitted[i]);sh.observe(emitted[i]);}
    }
    assert(sp==ar);
    std::puts("spec_hybrid_test: PASS (2/3/4-gram, ambiguity, bounded ring, greedy parity)");
}

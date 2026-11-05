#include "strata/core/spec_hybrid.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace strata::core {
namespace {
uint64_t hash_key(const int32_t* key, int n) {
    uint64_t h = 1469598103934665603ull;
    for(int i=0;i<n;++i){uint32_t v=uint32_t(key[i]);for(int b=0;b<4;++b){h^=(v>>(8*b))&255;h*=1099511628211ull;}}
    return h ? h : 1;
}
}

SyntaxBitmask::SyntaxBitmask() {
    words.fill(~0ull);
}

void SyntaxBitmask::apply_mask(float* logits, int vocab_size, int brace_depth, bool in_string) const {
    if (logits == nullptr || in_string || brace_depth <= 0) return;
    const size_t n = std::min<size_t>((size_t) vocab_size, kBits);
    const size_t n_words = (n + 63) / 64;

    for (size_t w = 0; w < n_words; ++w) {
        const uint64_t mask = words[w];
        if (mask == ~0ull) continue; // All 64 tokens valid
        const size_t base = w * 64;
        if (mask == 0ull) {
            const size_t limit = std::min<size_t>(base + 64, n);
            for (size_t i = base; i < limit; ++i) {
                logits[i] = -1e10f;
            }
            continue;
        }
        for (int b = 0; b < 64 && (base + b) < n; ++b) {
            if ((mask & (1ull << b)) == 0) {
                logits[base + b] = -1e10f;
            }
        }
    }
}

SpecHybrid::SpecHybrid(int window,size_t slots,size_t history):window_(window),history_cap_(history),ring_(history,0) {
    if(window<2||window>4||history<8)throw std::invalid_argument("spec hybrid window/history");
    size_t cap=1;while(cap<slots)cap<<=1;table_.resize(std::max<size_t>(16,cap));mask_=table_.size()-1;
}
void SpecHybrid::reset(){std::fill(table_.begin(),table_.end(),Slot{});history_size_=history_head_=clock_=0;last_order_=0;tracker_.reset();}
bool SpecHybrid::matches(const Slot& s,const int32_t* key)const{for(int i=0;i<s.order;++i)if(s.key[i]!=key[i])return false;return true;}
SpecHybrid::Slot* SpecHybrid::slot_for(const int32_t* key,int order,bool insert){
    const uint64_t h=hash_key(key,order);size_t p=h&mask_;
    for(size_t n=0;n<table_.size();++n,++p){p&=mask_;Slot&s=table_[p];if(!s.count){if(!insert)return nullptr;s.hash=h;s.order=order;for(int i=0;i<order;++i)s.key[i]=key[i];return &s;}if(s.hash==h&&s.order==order&&matches(s,key))return &s;}
    return nullptr;
}
void SpecHybrid::observe(const int32_t* tokens,size_t n){
    for(size_t j=0;j<n;++j){
        const int32_t tok=tokens[j];
        if (tok >= 0 && tok < 128) {
            tracker_.observe_char((char) tok);
        }
        for(int order=2;order<=4;++order){
            if(history_size_<size_t(order))continue;
            std::array<int32_t,4> key{};for(int k=0;k<order;++k)key[k]=at(history_size_-order+k);
            Slot* s=slot_for(key.data(),order,true);
            if(!s){Slot& victim=table_[(hash_key(key.data(),order)+clock_)&mask_];victim=Slot{};victim.hash=hash_key(key.data(),order);victim.order=order;std::copy(key.begin(),key.begin()+order,victim.key.begin());s=&victim;}
            s->stamp=++clock_;s->count++;
            bool found=false;for(uint8_t i=0;i<s->choices;++i)if(s->next[i]==tok){if(s->frequency[i]<65535)++s->frequency[i];found=true;break;}
            if(!found){if(s->choices<5){s->next[s->choices]=tok;s->frequency[s->choices++]=1;}else{s->choices=1;s->next[0]=tok;s->frequency[0]=1;}}
            if(history_size_>=size_t(order)){
                const size_t pos=history_size_-order;
                bool exists=false;for(uint8_t i=0;i<s->ways;++i)if(s->positions[i]==pos)exists=true;
                if(!exists){if(s->ways<4)++s->ways;for(int i=int(s->ways)-1;i>0;--i)s->positions[i]=s->positions[i-1];s->positions[0]=uint32_t(pos);}
            }
        }
        if(history_size_<history_cap_)++history_size_;else history_head_=(history_head_+1)%history_cap_;
        ring_[(history_head_+history_size_-1)%history_cap_]=tok;
    }
}
int SpecHybrid::propose(int max_drafts,int32_t* out){
    last_order_=0;const int limit=std::min(max_drafts,kMaxDrafts);if(limit<=0||history_size_<size_t(window_)){++misses_;return 0;}
    std::array<int32_t,4> key{};Slot* chosen=nullptr;int order=0;
    for(order=std::min(4,window_);order>=2;--order){
        if(history_size_<size_t(order))continue;
        for(int k=0;k<order;++k)key[k]=at(history_size_-order+k);
        Slot* s=slot_for(key.data(),order,false);if(!s||s->count<2||!s->choices)continue;
        int best=0;uint32_t total=0;for(uint8_t i=0;i<s->choices;++i){total+=s->frequency[i];if(s->frequency[i]>s->frequency[best])best=i;}
        if(s->frequency[best]*2>total){
            if (grammar_guidance_ && tracker_.should_filter() && !bitmask_.is_valid(s->next[best])) {
                continue; // Skip syntax-violating candidate under active delimiter scope
            }
            chosen=s;out[0]=s->next[best];last_order_=order;break;
        }
    }
    if(!chosen){++misses_;return 0;}
    int n=1;
    // Extend from an exact historical occurrence, not by chaining unrelated hash hits.
    for(uint8_t w=0;w<chosen->ways&&n<limit;++w){
        const size_t pos=chosen->positions[w];
        if(pos+order+limit>history_size_)continue;
        bool same=true;for(int k=0;k<order&&same;++k)same=at(pos+k)==key[k];
        if(!same||at(pos+order)!=out[0])continue;
        n=1;
        while(n<limit&&pos+order+n<history_size_){
            const int32_t cand = at(pos+order+n);
            if (grammar_guidance_ && tracker_.should_filter() && !bitmask_.is_valid(cand)) {
                break; // Stop extension before invalid token
            }
            out[n]=cand;++n;
        }
        if(n==limit)break;
    }
    ++hits_;return n;
}
int greedy_accept_prefix(const int32_t* drafts,int n,const int32_t* target,int32_t* emitted,int max_emitted){
    int a=0;while(a<n&&a<max_emitted&&drafts[a]==target[a]){emitted[a]=drafts[a];++a;}
    if(a<max_emitted)emitted[a]=target[a];
    return std::min(a+1,max_emitted);
}
} // namespace strata::core

#include "strata/core/radix_cache.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
using strata::core::RadixCache;
int main(){
    RadixCache c(4,5);
    const std::vector<int32_t> a{1,2,3,4,5,6,7,8};
    const std::vector<int32_t> b{1,2,3,4,9,10,11,12};
    const std::vector<int32_t> shorter{1,2,3};
    assert(c.insert(a,10)==10);
    assert(c.insert(b,20)==20);
    auto m=c.find_longest_prefix(b);assert(m.tokens==8&&m.payload==20&&m.node);
    m=c.find_longest_prefix({1,2,3,4,9,99});assert(m.tokens==5&&m.payload==0);
    assert(c.insert(shorter,30)==30);
    m=c.find_longest_prefix(a);assert(m.tokens==8&&m.payload==10);
    assert(c.used_blocks()<=4);
    assert(c.insert(a,99)==10); // duplicate is idempotent
    assert(c.pin(10));
    const std::vector<int32_t> d{1,2,3,4,13,14,15,16};
    assert(c.insert(d,40)==40); // pinned A cannot be evicted
    assert(c.find_longest_prefix(a).payload==10);
    c.unpin(10);
    c.find_longest_prefix(a); // refresh A
    const std::vector<int32_t> e{21,22,23,24};
    assert(c.insert(e,50)==50);
    assert(c.evictions()>0);
    assert(c.used_blocks()<=5);
    assert(c.find_longest_prefix(e).node!=nullptr);
    c.clear();assert(c.used_blocks()==0&&c.find_longest_prefix(a).tokens==0);
    RadixCache tiny(4,1);assert(tiny.insert(a,1)==0);assert(tiny.used_blocks()==0);
    RadixCache stress(8,32); uint32_t rng=12345;
    for(int i=0;i<2000;++i){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;std::vector<int32_t> t(1+rng%40);for(auto& v:t)v=int32_t(rng%13);stress.insert(t,uint64_t(i+1));assert(stress.used_blocks()<=32);auto m=stress.find_longest_prefix(t);assert(m.tokens<=t.size());if(i%97==0)stress.clear();}
    std::puts("radix_cache_test: PASS (LPM, split, branch, pin, LRU, capacity, stress)");
}

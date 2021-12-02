#include <tuple>
#include <utility>
#include <iostream>
#include <vector>
#include <compare>
#include <type_traits>
#include <limits>

#include "SpSpInst/SpSpInterface.hpp"
using namespace SpSpInst;
// print multi vectors
template<class...Ts, std::size_t... Is>
void printTuple(std::tuple<Ts*...> pointers,int l, std::index_sequence<Is...> s){
    for(int i=0; i<l; ++i){
        ((std::cout << std::get<Is>(pointers)[i] << "\t"),...);
        std::cout << std::endl;
    }
}

template<class...Ts>
using VRegTuple = std::tuple<VReg<Ts>...>;

inline const uint64_t longLimit = pack(Limit{cpu.logV,{OpSrc::B,Delta::NotEqual}});

template<class T, T val>
struct Dummy{};

enum class ImplSelect{
    SIMD, Scalar
};

template<class...Ts, std::size_t...Is>
inline void loadToReg_impl(std::index_sequence<Is...>, std::tuple<VReg<Ts>&...> regs, VecBool pred, std::tuple<Ts*...> ptrs, int offset){
    ((
        std::get<Is>(regs) = load(pred,std::get<Is>(ptrs)+offset)
    ),...);
}
template<class...Ts>
inline void loadToReg(std::tuple<VReg<Ts>&...> regs, VecBool pred, std::tuple<Ts*...> ptrs, int offset){
    loadToReg_impl(std::index_sequence_for<Ts...>{},regs, pred, ptrs,offset);
}

template<class...Ts>
inline void loadToReg_arglist(VReg<Ts>&... regs, VecBool pred, Ts*... ptrs, int offset){
    ((
        regs = load(pred,ptrs + offset)
    ),...);
    // loadToReg_impl(std::index_sequence_for<Ts...>{},regs, pred, ptrs,offset);
}


template<
    bool transparent, 
    class... Keys, std::size_t... IK,
    class... ValAs,std::size_t... IVa,
    class... ValBs,std::size_t... IVb,
    class... ValCs,std::size_t... IVc,
    class Callable
    >
int joinORVec_impl(
    Dummy<bool,transparent>,
    std::index_sequence<IK...>, 
    std::index_sequence<IVa...>,
    std::index_sequence<IVb...>,
    std::index_sequence<IVc...>,
    std::tuple<Keys*...> ka, std::tuple<ValAs*...> va, int lenA,
    std::tuple<Keys*...> kb, std::tuple<ValBs*...> vb, int lenB,
    std::tuple<Keys*...> kc, std::tuple<ValCs*...> vc,
    std::tuple<ValAs...> defaultA, std::tuple<ValBs...> defaultB,
    Callable op
){
    if constexpr (transparent){
        static_assert(std::is_same_v<std::tuple<ValAs...>,std::tuple<ValCs...>>);
        static_assert(std::is_same_v<std::tuple<ValBs...>,std::tuple<ValCs...>>);
    }
    using namespace SpSpInst;
    using VecKeyTuple = std::tuple<VReg<Keys>&...>;
    using VecValATuple = std::tuple<VReg<ValAs>&...>;
    using VecValBTuple = std::tuple<VReg<ValBs>&...>;
    using VecValCTuple = std::tuple<VReg<ValCs>&...>;

    static constexpr uint64_t getLimitOp2 = pack(GetLimitOp2{ForceEq::Yes,PolicyOR.eagerMask,{{Next::Epsilon,Next::Inf},{Next::Epsilon,Next::Inf}}}); 


    int pa,pb,pc;
    pa = pb = pc = 0;

    auto foo = [&] (
                    VReg<Keys>...keyARegs, VReg<Keys>...keyBRegs, VReg<ValAs>...valARegs, VReg<ValBs>...valBRegs,
                    VReg<ValAs>...valAGenRegs, VReg<ValBs>...valBGenRegs,VReg<ValCs>...valCGenRegs,
                        Keys*... ka_ptrs, ValAs*... va_ptrs, ValAs... defaultAs,
                        Keys*... kb_ptrs, ValBs*... vb_ptrs, ValBs... defaultBs,
                        Keys*... kc_ptrs, ValCs*... vc_ptrs
                         ){
        // VecKeyTuple keyA(keyARegs...);
        // VecKeyTuple keyB(keyBRegs...);
        // VecValATuple valA(valARegs...);
        // VecValBTuple valB(valBRegs...);
        // VecValATuple VCa(valAGenRegs...);
        // VecValBTuple VCb(valBGenRegs...);
        // VecValCTuple VCc(valCGenRegs...);

        while(pa < lenA && pb < lenB){
            VecBool predA = whilelt(pa,lenA);
            VecBool predB = whilelt(pb,lenB);

            loadToReg_arglist<Keys...>(keyARegs...,predA,ka_ptrs...,pa);
            loadToReg_arglist<Keys...>(keyBRegs...,predB,kb_ptrs...,pb);
            loadToReg_arglist<ValAs...>(valARegs...,predA,va_ptrs...,pa);
            loadToReg_arglist<ValBs...>(valBRegs...,predB,vb_ptrs...,pb);
            // loadToReg(keyA,predA,ka,pa);
            // loadToReg(keyB,predB,kb,pb);
            // loadToReg(valA,predA,va,pa);
            // loadToReg(valB,predB,vb,pb);

            VBigCmp bigCmp = InitBigCmp(longLimit,predA,predB);
            ((bigCmp=KeyCombine<DefMethod<Keys>>(bigCmp,keyARegs,keyBRegs)),...);
            VMatRes matRes = Match(bigCmp,PolicyOR.policyMask.A,PolicyOR.policyMask.B);
            uint64_t newLimit = GetLimit(bigCmp,PolicyOR.simPolicyMask,getLimitOp2);
            Limit unpackLimit = unpack<Limit>(newLimit);
            int genC = unpackLimit.generate.A;

            auto processPart = [&]<LRPart part> (Dummy<LRPart,part>){
                auto extraOffset = part == LRPart::Left? 0 : cpu.v;
                VecBool predC = whilelt(extraOffset,genC);
                
                (
                    (
                        store(predC,kc_ptrs+pc + extraOffset,
                            simd_or(predC,
                                SEPermute<SEPart{OpSrc::A,part}>(matRes,keyARegs,SEPair{Keys(0),Keys(0)}),
                                SEPermute<SEPart{OpSrc::B,part}>(matRes,keyBRegs,SEPair{Keys(0),Keys(0)})
                            )
                        )
                    ),...
                );

                // ((
                //     valAGenRegs = SEPermute<SEPart{OpSrc::A,part}>(matRes,std::get<IVa>(valA),SEPair{ValAs(0),ValAs(0)})
                //     ),...);
                // ((
                //     valBGenRegs = SEPermute<SEPart{OpSrc::B,part}>(matRes,std::get<IVb>(valB),SEPair{ValBs(0),ValBs(0)})
                //     ),...);
                

                ((
                    valAGenRegs = SEPermute<SEPart{OpSrc::A,part}>(matRes,valARegs,SEPair{std::get<IVa>(defaultA),ValAs(0)})
                    ),...);
                ((
                    valBGenRegs = SEPermute<SEPart{OpSrc::B,part}>(matRes,valBRegs,SEPair{std::get<IVb>(defaultB),ValBs(0)})
                    ),...);

                // VecValCTuple VCc = op(predC,VCa,VCb);
                op(predC,
                    valAGenRegs...,
                    valBGenRegs...,
                    valCGenRegs...);
                // op(predC,
                //     std::tuple<VReg<ValAs>&...>(std::get<IVa>(VCa)...),
                //     std::tuple<VReg<ValBs>&...>(std::get<IVb>(VCb)...),
                //     std::tuple<VReg<ValCs>&...>(std::get<IVc>(VCc)...));

                (
                    (
                        store(predC,vc_ptrs+pc + extraOffset,valCGenRegs)
                    ),...
                );

            };

            processPart(Dummy<LRPart,LRPart::Left>{});
            if(genC > cpu.v){
                processPart(Dummy<LRPart,LRPart::Right>{});
            }

            pa += unpackLimit.consume.A;
            pb += unpackLimit.consume.B;
            pc += genC;
        }

        if(pa < lenA){
            printf("Continue pa<lenA\n");
            auto buffer_pa = pa;
            auto buffer_pc = pc;

            while(pa < lenA){
                VecBool predC = whilelt(pa,lenA);
                ((store(predC, kc_ptrs + pc,
                    load(predC, ka_ptrs + pa))),...);
                if constexpr (transparent){
                    ((store(predC, vc_ptrs + pc,
                        load(predC, va_ptrs + pa))),...);
                } else {
                // VecValATuple VCa = std::make_tuple(load(predC,std::get<IVa>(va)+pa)...);
                // VecValBTuple VCb = std::make_tuple(dup(std::get<IVb>(defaultB))...);

                loadToReg_arglist<ValAs...>(valAGenRegs...,predC,va_ptrs...,pa);
                ((
                    valBGenRegs = dup(std::get<IVb>(defaultB))
                    ),...);

                // VecValCTuple VCc = op(predC,VCa,VCb);
                op(predC,
                    valAGenRegs...,
                    valBGenRegs...,
                    valCGenRegs...);
                // op(pred,
                //     std::tuple<VReg<ValAs>&...>(std::get<IVa>(VCa)...),
                //     std::tuple<VReg<ValBs>&...>(std::get<IVb>(VCb)...),
                //     std::tuple<VReg<ValCs>&...>(std::get<IVc>(VCc)...));

                    ((store(predC, vc_ptrs + pc, valCGenRegs )),...);
                }
                pa += cpu.v;
                pc += cpu.v;
            }
            pc = buffer_pc + (lenA - buffer_pa);
        }
        if(pb < lenB){
            printf("Continue pb<lenB\n");
            auto buffer_pb = pb;
            auto buffer_pc = pc;

            while(pb < lenB){
                printf("pb = %d, lenB = %d, pc = %d\n",pb,lenB,pc);
                VecBool predC = whilelt(pb,lenB);
                ((store(predC, kc_ptrs + pc,
                    load(predC, kb_ptrs + pb))),...);
                if constexpr (transparent){
                    ((store(predC, vc_ptrs + pc,
                        load(predC, vb_ptrs + pb))),...);
                } else {
                ((
                   valAGenRegs = dup(std::get<IVa>(defaultA))
                    ),...);
                loadToReg_arglist<ValBs...>(valBGenRegs...,predC,vb_ptrs...,pb);
                op(predC,
                    valAGenRegs...,
                    valBGenRegs...,
                    valCGenRegs...);
                // op(pred,
                //     std::tuple<VReg<ValAs>&...>(std::get<IVa>(VCa)...),
                //     std::tuple<VReg<ValBs>&...>(std::get<IVb>(VCb)...),
                //     std::tuple<VReg<ValCs>&...>(std::get<IVc>(VCc)...));
                    ((store(predC, vc_ptrs + pc, valCGenRegs)),...);
                }
                pb += cpu.v;
                pc += cpu.v;
            }
            pc = buffer_pc + (lenB - buffer_pb);
        }
    };
    foo(
        VReg<Keys>{}...,VReg<Keys>{}...,VReg<ValAs>{}...,VReg<ValBs>{}...,
        VReg<ValAs>{}...,VReg<ValBs>{}...,VReg<ValCs>{}...,
        std::get<IK>(ka)..., std::get<IVa>(va)..., std::get<IVa>(defaultA)...,
        std::get<IK>(kb)..., std::get<IVb>(vb)..., std::get<IVb>(defaultB)...,
        std::get<IK>(kc)..., std::get<IVc>(vc)...
        );
    return pc;
}

template<
    bool transparent,
    class... Keys, 
    class... ValAs,
    class... ValBs,
    class... ValCs,
    class Callable
    >
int joinORVec(
    Dummy<bool,transparent>,
    std::tuple<Keys*...> ka, std::tuple<ValAs*...> va, int lenA,
    std::tuple<Keys*...> kb, std::tuple<ValBs*...> vb, int lenB,
    std::tuple<Keys*...> kc, std::tuple<ValCs*...> vc,
    std::tuple<ValAs...> defaultA, std::tuple<ValBs...> defaultB,
    Callable op
){
    return joinORVec_impl(
        Dummy<bool,transparent>{},
        std::index_sequence_for<Keys...>{},
        std::index_sequence_for<ValAs...>{},
        std::index_sequence_for<ValBs...>{},
        std::index_sequence_for<ValCs...>{},
        ka,va,lenA,
        kb,vb,lenB,
        kc,vc,
        defaultA, defaultB,
        op
    );
}

template<
    bool transparent,
    class... Keys, std::size_t... IK,
    class... ValAs,std::size_t... IVa,
    class... ValBs,std::size_t... IVb,
    class... ValCs,std::size_t... IVc,
    class Callable
    >
int joinOR_impl(
    Dummy<bool,transparent>,
    std::index_sequence<IK...>, 
    std::index_sequence<IVa...>,
    std::index_sequence<IVb...>,
    std::index_sequence<IVc...>,
    std::tuple<Keys*...> ka, std::tuple<ValAs*...> va, int lenA,
    std::tuple<Keys*...> kb, std::tuple<ValBs*...> vb, int lenB,
    std::tuple<Keys*...> kc, std::tuple<ValCs*...> vc,
    std::tuple<ValAs...> defaultA, std::tuple<ValBs...> defaultB,
    Callable op
){
    if constexpr (transparent){
        static_assert(std::is_same_v<std::tuple<ValAs...>,std::tuple<ValCs...>>);
        static_assert(std::is_same_v<std::tuple<ValBs...>,std::tuple<ValCs...>>);
    }
    int pa,pb,pc;
    pa = pb = pc = 0;

    while(pa < lenA && pb < lenB){
        auto keyA = std::make_tuple(std::get<IK>(ka)[pa]...);
        auto keyB = std::make_tuple(std::get<IK>(kb)[pb]...);
        auto cmp = keyA <=> keyB;

        std::tuple<ValAs...> elemA;
        std::tuple<ValBs...> elemB;
        std::tuple<Keys...> keyC;
        std::tuple<ValCs...> resultC;

        if(std::is_eq(cmp)){ // keyA == keyB
            keyC = keyA;
            elemA = std::make_tuple(std::get<IVa>(va)[pa]...);
            elemB = std::make_tuple(std::get<IVb>(vb)[pb]...);
            pa ++;
            pb ++;
        } else if(std::is_lt(cmp)){ // keyA < keyB
            keyC = keyA;
            elemA = std::make_tuple(std::get<IVa>(va)[pa]...);
            elemB = defaultB;
            pa ++;      
        } else { // keyA > keyB
            keyC = keyB;
            elemA = defaultA;
            elemB = std::make_tuple(std::get<IVb>(vb)[pb]...);
            pb ++;
        }
        ((std::get<IK>(kc)[pc]=std::get<IK>(keyC)),...);
        resultC = op(elemA,elemB);
        ((std::get<IVc>(vc)[pc]=std::get<IVc>(resultC)),...);
        pc ++;
    }
    if(pa < lenA){
        ((std::copy(std::get<IK>(ka) + pa, std::get<IK>(ka) + lenA, std::get<IK>(kc) + pc)), ...);
        if constexpr (transparent){
            ((std::copy(std::get<IVc>(va) + pa, std::get<IVc>(va) + lenA, std::get<IVc>(vc) + pc)),...);
            pc += lenA - pa;
        } else {
            while(pa < lenA){
                auto elemA = std::make_tuple(std::get<IVa>(va)[pa]...);
                auto elemB = defaultB;
                auto resultC = op(elemA,elemB);
                ((std::get<IVc>(vc)[pc]=std::get<IVc>(resultC)),...);
                pa ++;
                pc ++;
            }
        }
    }
    if(pb < lenB){
        ((std::copy(std::get<IK>(kb) + pb, std::get<IK>(kb) + lenB, std::get<IK>(kc) + pc)), ...);
        if constexpr (transparent){
            ((std::copy(std::get<IVc>(vb) + pb, std::get<IVc>(vb) + lenB, std::get<IVc>(vc) + pc)),...);
            pc += lenB - pb;
        } else {
            while(pb < lenB){
                auto elemA = defaultA;
                auto elemB = std::make_tuple(std::get<IVa>(vb)[pb]...);
                auto resultC = op(elemA,elemB);
                ((std::get<IVc>(vc)[pc]=std::get<IVc>(resultC)),...);
                pb ++;
                pc ++;
            }
        }
    }
    return pc;
}

template<
    bool transparent,
    class... Keys,
    class... ValAs,
    class... ValBs,
    class... ValCs,
    class Callable
    >
int joinOR(
    Dummy<bool,transparent>,
    std::tuple<Keys*...> ka, std::tuple<ValAs*...> va, int lenA,
    std::tuple<Keys*...> kb, std::tuple<ValBs*...> vb, int lenB,
    std::tuple<Keys*...> kc, std::tuple<ValCs*...> vc,
    std::tuple<ValAs...> defaultA, std::tuple<ValBs...> defaultB,
    Callable op
){
    return joinOR_impl(
        Dummy<bool,transparent>{},
        std::index_sequence_for<Keys...>{},
        std::index_sequence_for<ValAs...>{},
        std::index_sequence_for<ValBs...>{},
        std::index_sequence_for<ValCs...>{},
        ka,va,lenA,
        kb,vb,lenB,
        kc,vc,
        defaultA, defaultB,
        op
    );
}

// template<
//     ImplSelect impl,
//     bool transparent,
//     class... Keys, 
//     class... ValAs,
//     class... ValBs,
//     class... ValCs,
//     class Callable
//     >
// int joinOR(
//     Dummy<ImplSelect,impl>,
//     Dummy<bool,transparent>,
//     std::tuple<Keys*...> ka, std::tuple<ValAs*...> va, int lenA,
//     std::tuple<Keys*...> kb, std::tuple<ValBs*...> vb, int lenB,
//     std::tuple<Keys*...> kc, std::tuple<ValCs*...> vc,
//     std::tuple<ValAs...> defaultA, std::tuple<ValBs...> defaultB,
//     Callable op
// ){
//     if constexpr(impl == ImplSelect::SIMD){
//         return joinORVec_impl(
//             Dummy<bool,transparent>{},
//             std::index_sequence_for<Keys...>{},
//             std::index_sequence_for<ValAs...>{},
//             std::index_sequence_for<ValBs...>{},
//             std::index_sequence_for<ValCs...>{},
//             ka,va,lenA,
//             kb,vb,lenB,
//             kc,vc,
//             defaultA, defaultB,
//             op
//         );
//     } else {
//         return joinOR_impl(
//             Dummy<bool,transparent>{},
//             std::index_sequence_for<Keys...>{},
//             std::index_sequence_for<ValAs...>{},
//             std::index_sequence_for<ValBs...>{},
//             std::index_sequence_for<ValCs...>{},
//             ka,va,lenA,
//             kb,vb,lenB,
//             kc,vc,
//             defaultA, defaultB,
//             op
//         );
//     }
// }




template<
    class... Keys, std::size_t... IK,
    class... ValAs,std::size_t... IVa,
    class... ValBs,std::size_t... IVb,
    class... ValCs,std::size_t... IVc,
    class Callable
    >
int joinANDVec_impl(
    std::index_sequence<IK...>, 
    std::index_sequence<IVa...>,
    std::index_sequence<IVb...>,
    std::index_sequence<IVc...>,
    std::tuple<Keys*...> ka, std::tuple<ValAs*...> va, int lenA,
    std::tuple<Keys*...> kb, std::tuple<ValBs*...> vb, int lenB,
    std::tuple<Keys*...> kc, std::tuple<ValCs*...> vc,
    Callable op
){
    using namespace SpSpInst;
    using VecKeyTuple = std::tuple<VReg<Keys>&...>;
    using VecValATuple = std::tuple<VReg<ValAs>&...>;
    using VecValBTuple = std::tuple<VReg<ValBs>&...>;
    using VecValCTuple = std::tuple<VReg<ValCs>&...>;

    static constexpr uint64_t getLimitOp2 = pack(GetLimitOp2{ForceEq::Yes,PolicyAND.eagerMask,{{Next::Epsilon,Next::Inf},{Next::Epsilon,Next::Inf}}}); 

    int pa,pb,pc;
    pa = pb = pc = 0;

    auto foo = [&] (VReg<Keys>...keyARegs, VReg<Keys>...keyBRegs, VReg<ValAs>...valARegs, VReg<ValBs>...valBRegs,
                    VReg<ValAs>...valAGenRegs, VReg<ValBs>...valBGenRegs,VReg<ValCs>...valCGenRegs,
                         Keys*... ka_ptrs, ValAs*... va_ptrs, 
                        Keys*... kb_ptrs, ValBs*... vb_ptrs, 
                        Keys*... kc_ptrs, ValCs*... vc_ptrs
                     ){
        VecKeyTuple keyA(keyARegs...);
        VecKeyTuple keyB(keyBRegs...);
        VecValATuple valA(valARegs...);
        VecValBTuple valB(valBRegs...);
        VecValATuple VCa(valAGenRegs...);
        VecValBTuple VCb(valBGenRegs...);
        VecValCTuple VCc(valCGenRegs...);

        while(pa < lenA && pb < lenB){
            VecBool predA = whilelt(pa,lenA);
            VecBool predB = whilelt(pb,lenB);

            loadToReg_arglist<Keys...>(keyARegs...,predA,ka_ptrs...,pa);
            loadToReg_arglist<Keys...>(keyBRegs...,predB,kb_ptrs...,pb);
            loadToReg_arglist<ValAs...>(valARegs...,predA,va_ptrs...,pa);
            loadToReg_arglist<ValBs...>(valBRegs...,predB,vb_ptrs...,pb);


            VBigCmp bigCmp = InitBigCmp(longLimit,predA,predB);
            ((bigCmp=KeyCombine<DefMethod<Keys>>(bigCmp,std::get<IK>(keyA),std::get<IK>(keyB))),...);
            VMatRes matRes = Match(bigCmp,PolicyAND.policyMask.A,PolicyAND.policyMask.B);
            uint64_t newLimit = GetLimit(bigCmp,PolicyAND.simPolicyMask,getLimitOp2);
            Limit unpackLimit = unpack<Limit>(newLimit);
            int genC = unpackLimit.generate.A;

            auto processPart = [&]<LRPart part> (Dummy<LRPart,part>){
                auto extraOffset = part == LRPart::Left? 0 : cpu.v;
                VecBool predC = whilelt(extraOffset,genC);
                
                (
                    (
                        store(predC,std::get<IK>(kc)+pc + extraOffset,
                                SEPermute<SEPart{OpSrc::A,part}>(matRes,std::get<IK>(keyA),SEPair{Keys(0),Keys(0)})   
                        )
                    ),...
                );

                ((
                    valAGenRegs = SEPermute<SEPart{OpSrc::A,part}>(matRes,valARegs,SEPair{ValAs(0),ValAs(0)})
                    ),...);
                ((
                    valBGenRegs = SEPermute<SEPart{OpSrc::B,part}>(matRes,valBRegs,SEPair{ValAs(0),ValBs(0)})
                    ),...);

                // VecValCTuple VCc = op(predC,VCa,VCb);
                op(predC,
                    valAGenRegs...,
                    valBGenRegs...,
                    valCGenRegs...);

                (
                    (
                        store(predC,vc_ptrs+pc + extraOffset,valCGenRegs)
                    ),...
                );

            };

            processPart(Dummy<LRPart,LRPart::Left>{});


            pa += unpackLimit.consume.A;
            pb += unpackLimit.consume.B;
            pc += genC;
        }
    };
    foo(
        VReg<Keys>{}...,VReg<Keys>{}...,VReg<ValAs>{}...,VReg<ValBs>{}...,
        VReg<ValAs>{}...,VReg<ValBs>{}...,VReg<ValCs>{}...,
        std::get<IK>(ka)..., std::get<IVa>(va)...,
        std::get<IK>(kb)..., std::get<IVb>(vb)...,
        std::get<IK>(kc)..., std::get<IVc>(vc)...
        );

    return pc;
}

template<
    class... Keys, 
    class... ValAs,
    class... ValBs,
    class... ValCs,
    class Callable
    >
int joinANDVec(
    std::tuple<Keys*...> ka, std::tuple<ValAs*...> va, int lenA,
    std::tuple<Keys*...> kb, std::tuple<ValBs*...> vb, int lenB,
    std::tuple<Keys*...> kc, std::tuple<ValCs*...> vc,
    Callable op
){
    return joinANDVec_impl(
        std::index_sequence_for<Keys...>{},
        std::index_sequence_for<ValAs...>{},
        std::index_sequence_for<ValBs...>{},
        std::index_sequence_for<ValCs...>{},
        ka,va,lenA,
        kb,vb,lenB,
        kc,vc,
        op
    );
}



template<
    class... Keys, std::size_t... IK,
    class... ValAs,std::size_t... IVa,
    class... ValBs,std::size_t... IVb,
    class... ValCs,std::size_t... IVc,
    class Callable
    >
int joinAND_impl(
    std::index_sequence<IK...>, 
    std::index_sequence<IVa...>,
    std::index_sequence<IVb...>,
    std::index_sequence<IVc...>,
    std::tuple<Keys*...> ka, std::tuple<ValAs*...> va, int lenA,
    std::tuple<Keys*...> kb, std::tuple<ValBs*...> vb, int lenB,
    std::tuple<Keys*...> kc, std::tuple<ValCs*...> vc,
    Callable op
){
    int pa,pb,pc;
    pa = pb = pc = 0;
    while(pa < lenA && pb < lenB){
        auto keyA = std::make_tuple(std::get<IK>(ka)[pa]...);
        auto keyB = std::make_tuple(std::get<IK>(kb)[pb]...);
        auto cmp = keyA <=> keyB;

        if(std::is_eq(cmp)){ // keyA == keyB
            ((std::get<IK>(kc)[pc]=std::get<IK>(keyA)),...);
            std::tuple<ValAs...> elemA(std::get<IVa>(va)[pa]...);
            std::tuple<ValBs...> elemB(std::get<IVb>(vb)[pb]...);
            std::tuple<ValCs...> resultC = op(elemA,elemB);
            ((std::get<IVc>(vc)[pc]=std::get<IVc>(resultC)),...);
            pa ++;
            pb ++;
            pc ++;
        } else if(std::is_lt(cmp)){ // keyA < keyB
            pa ++;      
        } else { // keyA > keyB
            pb ++;
        }
    }

    return pc;
}

template<
    class... Keys,
    class... ValAs,
    class... ValBs,
    class... ValCs,
    class Callable
    >
int joinAND(
    std::tuple<Keys*...> ka, std::tuple<ValAs*...> va, int lenA,
    std::tuple<Keys*...> kb, std::tuple<ValBs*...> vb, int lenB,
    std::tuple<Keys*...> kc, std::tuple<ValCs*...> vc,
    Callable op
){
    return joinAND_impl(
        std::index_sequence_for<Keys...>{},
        std::index_sequence_for<ValAs...>{},
        std::index_sequence_for<ValBs...>{},
        std::index_sequence_for<ValCs...>{},
        ka,va,lenA,
        kb,vb,lenB,
        kc,vc,
        op
    );
}

// template<
//     ImplSelect impl,
//     class... Keys,
//     class... ValAs,
//     class... ValBs,
//     class... ValCs,
//     class Callable
//     >
// int joinAND(
//     Dummy<ImplSelect,impl>,
//     std::tuple<Keys*...> ka, std::tuple<ValAs*...> va, int lenA,
//     std::tuple<Keys*...> kb, std::tuple<ValBs*...> vb, int lenB,
//     std::tuple<Keys*...> kc, std::tuple<ValCs*...> vc,
//     Callable op
// ){
//     if constexpr(impl==ImplSelect::SIMD){
//         return joinANDVec_impl(
//             std::index_sequence_for<Keys...>{},
//             std::index_sequence_for<ValAs...>{},
//             std::index_sequence_for<ValBs...>{},
//             std::index_sequence_for<ValCs...>{},
//             ka,va,lenA,
//             kb,vb,lenB,
//             kc,vc,
//             op
//         );
//     } else {
//         return joinAND_impl(
//             std::index_sequence_for<Keys...>{},
//             std::index_sequence_for<ValAs...>{},
//             std::index_sequence_for<ValBs...>{},
//             std::index_sequence_for<ValCs...>{},
//             ka,va,lenA,
//             kb,vb,lenB,
//             kc,vc,
//             op
//         );
//     }
// }

template<class Array, std::size_t...I>
inline auto _arrayNtoTuple_impl(Array arr, std::index_sequence<I...>){
    return std::make_tuple(arr[I]...);
}

template<int N, class Array>
inline auto makeTFA(Array && arr){
    return _arrayNtoTuple_impl(arr,std::make_index_sequence<N>{});
}



// template<ImplSelect impl, std::size_t...Is>
// int sparseTensorAdd_Impl(
//     Dummy<ImplSelect,impl>,
//     std::index_sequence<Is...>,
//     std::vector<int*> ka, float *va, int lenA,
//     std::vector<int*> kb, float *vb, int lenB,
//     std::vector<int*> kc, float *vc
// ){
//     if constexpr (impl==ImplSelect::Scalar){
//         return joinOR(
//             Dummy<bool,true>{},
//             std::make_tuple(ka[Is]...), std::make_tuple(va),lenA,
//             std::make_tuple(kb[Is]...), std::make_tuple(vb),lenB,
//             std::make_tuple(kc[Is]...), std::make_tuple(vc),
//             std::make_tuple<float>(0),std::make_tuple<float>(0),
//             [](std::tuple<float> ta, std::tuple<float> tb){
//                 return std::make_tuple(std::get<0>(ta) + std::get<0>(tb));
//             }
//         );

//     }
// }
template<ImplSelect impl, int N>
int sparseTensorAdd(
    Dummy<ImplSelect,impl>,
    Dummy<int,N>,
    std::vector<int*> ka, float *va, int lenA,
    std::vector<int*> kb, float *vb, int lenB,
    std::vector<int*> kc, float *vc
){
    if constexpr (impl==ImplSelect::Scalar){
        return joinOR(
            Dummy<bool,true>{},
            makeTFA<N>(ka), std::make_tuple(va),lenA,
            makeTFA<N>(kb), std::make_tuple(vb),lenB,
            makeTFA<N>(kc), std::make_tuple(vc),
            std::make_tuple<float>(0),std::make_tuple<float>(0),
            [](std::tuple<float> ta, std::tuple<float> tb){
                return std::make_tuple(std::get<0>(ta) + std::get<0>(tb));
            }
        );
    } else {
        return joinORVec(
            Dummy<bool,true>{},
            makeTFA<N>(ka), std::make_tuple(va),lenA,
            makeTFA<N>(kb), std::make_tuple(vb),lenB,
            makeTFA<N>(kc), std::make_tuple(vc),
            std::make_tuple<float>(0),std::make_tuple<float>(0),
            [](VecBool pred, VReg<float>& ta, VReg<float>& tb,VReg<float>& tc){
                tc=simd_add(pred, ta,tb);
                // std::get<0>(tc)=(simd_add(pred, std::get<0>(ta),std::get<0>(tb)));
            }
            // [](VecBool pred, std::tuple<VReg<float>&> ta, std::tuple<VReg<float>&> tb,std::tuple<VReg<float>&> tc){
            //     std::get<0>(tc)=(simd_add(pred, std::get<0>(ta),std::get<0>(tb)));
            // }
        );
    }
}

template<ImplSelect impl>
int selectSorterPath(
    Dummy<ImplSelect,impl>,
    int* ka, float *va, int lenA,
    int* kb, float *vb, int lenB,
    int* kc, int* node, float *vc,
    int nodeA, float distA,
    int nodeB, float distB
){
    float floatMax = std::numeric_limits<float>::max();
    if constexpr (impl==ImplSelect::Scalar){
        return joinOR(
            Dummy<bool,false>{},
            std::make_tuple(ka), std::make_tuple(va),lenA,
            std::make_tuple(kb), std::make_tuple(vb),lenB,
            std::make_tuple(kc), std::make_tuple(node,vc),
            std::tuple<float>(floatMax),std::tuple<float>(floatMax),
            [nodeA, distA, nodeB, distB](std::tuple<float> ta, std::tuple<float> tb){
                auto [va] = ta;
                auto [vb] = tb;
                float distAx = distA + va;
                float distBx = distB + vb;
                bool selectA = distAx <= distBx;
                return selectA? std::make_tuple(nodeA,distAx) : std::make_tuple(nodeB,distBx);
            }
        );
    } else {
        return joinORVec(
            Dummy<bool,false>{},
            std::make_tuple(ka), std::make_tuple(va),lenA,
            std::make_tuple(kb), std::make_tuple(vb),lenB,
            std::make_tuple(kc), std::make_tuple(node,vc),
            std::tuple<float>(floatMax),std::tuple<float>(floatMax),
            [nodeA, distA, nodeB, distB] (VecBool pred, VReg<float>& va, VReg<float>& vb,
                VReg<int>& c_node,VReg<float>& c_Dist){
                auto distAx = simd_add_vs(pred,va,distA);
                auto distBx = simd_add_vs(pred,vb,distB);
                VecBool selectA = simd_cmple(pred,distAx, distBx);
                c_node = simd_select(selectA, dup(nodeA), dup(nodeB));
                c_Dist = simd_select(selectA, distAx, distBx);
            }
        );
    }
}


template<ImplSelect impl, int N>
int sparseTensorMul(
    Dummy<ImplSelect,impl>,
    Dummy<int,N>,
    std::vector<int*> ka, float *va, int lenA,
    std::vector<int*> kb, float *vb, int lenB,
    std::vector<int*> kc, float *vc
){
    if constexpr (impl==ImplSelect::Scalar){
        return joinAND(
            makeTFA<N>(ka), std::make_tuple(va),lenA,
            makeTFA<N>(kb), std::make_tuple(vb),lenB,
            makeTFA<N>(kc), std::make_tuple(vc),
            [](std::tuple<float> ta, std::tuple<float> tb){
                return std::make_tuple(std::get<0>(ta) * std::get<0>(tb));
            }
        );
    } else {
        return joinANDVec(
            makeTFA<N>(ka), std::make_tuple(va),lenA,
            makeTFA<N>(kb), std::make_tuple(vb),lenB,
            makeTFA<N>(kc), std::make_tuple(vc),
            [](VecBool pred, VReg<float>& ta, VReg<float>& tb, VReg<float>& tc){
                tc = simd_mul(pred,ta,tb);
                // std::get<0>(tc) = simd_mul(pred, std::get<0>(ta),std::get<0>(tb));
                // return std::tuple<VReg<float>&>(simd_mul(pred, std::get<0>(ta),std::get<0>(tb)));
            }
        );
    }
}

template<ImplSelect impl, int N>
int sparseTensorMulComplex(
    Dummy<ImplSelect,impl>,
    Dummy<int,N>,
    std::vector<int*> ka, std::tuple<float*,float*> va, int lenA,
    std::vector<int*> kb, std::tuple<float*,float*> vb, int lenB,
    std::vector<int*> kc, std::tuple<float*,float*> vc
){
    if constexpr (impl==ImplSelect::Scalar){
        return joinAND(
            makeTFA<N>(ka), va,lenA,
            makeTFA<N>(kb), vb,lenB,
            makeTFA<N>(kc), vc,
            [](std::tuple<float,float> ta, std::tuple<float,float> tb){
                auto [a_re,a_im] = ta;
                auto [b_re,b_im] = tb;
                return std::make_tuple(
                    a_re * b_re - a_im * b_im,
                    a_re * b_im + a_im * b_re
                    );
            }
        );
    } else {
        return joinANDVec(
            makeTFA<N>(ka), va,lenA,
            makeTFA<N>(kb), vb,lenB,
            makeTFA<N>(kc), vc,
            [](VecBool pred, VReg<float>& a_re, VReg<float>& a_im, VReg<float>& b_re, VReg<float>& b_im,
                                VReg<float>& c_re, VReg<float>& c_im){
                c_re = simd_sub(pred,simd_mul(pred,a_re,b_re),simd_mul(pred,a_im,b_im));
                c_im = simd_add(pred,simd_mul(pred,a_re,b_im),simd_mul(pred,a_im,b_re));
            }
        );
    }
}


// int test(){
//     VReg<int> a,b,c,d;
//     std::tuple<VReg<int>&,VReg<int>&,VReg<int>&,VReg<int>&> s = {a,b,c,d};
//     VecBool pred;
//     std::get<0>(s) = simd_add(pred,std::get<1>(s),std::get<2>(s));
//     a = VReg<int>{};

//     return 0;

// }



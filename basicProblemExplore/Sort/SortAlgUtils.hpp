#pragma once
#include <stdint.h>
#include <bit>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <unordered_set>
#include "../SimpleBenchmarking/BenchMarking.hpp"
#include "SpSpInst/SpSpInterface.hpp"

namespace SortAlg
{
    using namespace SpSpInst;

    // auto policyFactory =
    // SpSpPolicyFactory::PolicyFactory(SpSPPredefPolicy::UniqueOR); auto policyMask
    // = policyFactory.policyMask();

    enum struct SortState : uint8_t
    {
        Init,
        LeftIssued,
        RightIssued
    };

    struct Task
    {
        uint8_t srcBuf, destBuf;
        SortState state;
        int start, len;
    };

    constexpr uint64_t shortLimit = pack(Limit{0, {OpSrc::B, Delta::NotEqual}});

    // constexpr auto sortFactory =
    //     SpSpPolicyFactory::PolicyFactory(SpSPPredefPolicy::SORT);
    // constexpr auto sortSimMask = sortFactory.simPolicyMask();
    // constexpr auto sortEagerMask = sortFactory.eagerMask();
    constexpr auto sortOp2 =
        pack(GetLimitOp2{ForceEq::Yes,
                         PolicySORT.eagerMask,
                         {{Next::Same, Next::Inf}, {Next::Same, Next::Inf}}});
    const uint64_t longLimit = pack(Limit{cpu.logV, {OpSrc::B, Delta::NotEqual}});

    // Reqire: len > cpu.v, then both left and right will not be zero.
    // left + right = len, left = n * cpu.v,  (len - modV(len))/2 <= left <= (len -
    // modV(len)+cpu.v)/2 (len + modV(len))/2 >= right >= (len + modV(len) -
    // cpu.v)/2 Usually only one choice. When two boundary are k*cpu.v and
    // (k+1)*cpu.v, then left k*cpu.v (2kv ~ 2kv+v-1, left = kv)
    inline std::pair<int, int> getChildLen(int len)
    {
        int x = len >> cpu.logV2;          // x = (len - modV(len)) >> cpu.v
        int xv2 = x >> 1;                  // xv2 = (x - mod2(x)) >> 1
        int left = (x - xv2) << cpu.logV2; // (x - xv2) = (x + mod2(x)) >> 1,
        // left =   ( ( (len - modV(len)) >> cpu.v + mod2( (len - modV(len)) >> cpu.v
        // ) >> 1 << cpu.v left  << 1 =  (len - modV(len))  +  mod2((len - modV(len))
        // >> cpu.v)  << cpu.v left << 1 >= len - modV(len) left << 1 <= len -
        // modV(len) + cpu.v

        int right = len - left;
        return std::make_pair(left, right);
    }

    enum class CommandOp
    {
        Exit,
        ShortSort,
        Merge
    };
    // struct Command{
    //     CommandOp op;
    //     int start, mid ,end;
    // };

    struct MergeSortStateMachine
    {
        std::unique_ptr<Task[]> ts;
        Task *top, *bottom;

        CommandOp op = CommandOp::Exit;
        int start = 0, left = 0, right = 0, total = 0;
        uint8_t srcBuf = 0, destBuf = 1;

        MergeSortStateMachine(int len)
        {
            ts = std::make_unique<Task[]>(64);
            top = bottom = ts.get();

            bottom->srcBuf = 0;
            bottom->destBuf = 0;
            bottom->start = 0;
            bottom->len = len;
            bottom->state = SortState::Init;

            next();
        }

        void next()
        {
            while (true)
            {
                if (top < bottom)
                {
                    op = CommandOp::Exit;
                    return;
                }
                switch (top->state)
                {
                case SortState::Init:
                    if (top->len <= cpu.v2)
                    {
                        op = CommandOp::ShortSort;
                        start = top->start;
                        total = top->len;
                        srcBuf = top->srcBuf;
                        destBuf = top->destBuf;

                        // shortSort<method, T>(buf[top->srcBuf] + top->start,
                        //                     buf[top->destBuf] + top->start, top->len);
                        --top;
                        return;
                        // continue;
                    }
                    else
                    {
                        auto newTop = top + 1;
                        auto [left, right] = getChildLen(top->len);

                        newTop->srcBuf = top->srcBuf;
                        newTop->destBuf = 1 - top->destBuf;
                        newTop->start = top->start;
                        newTop->len = left;
                        newTop->state = SortState::Init;
                        top->state = SortState::LeftIssued;
                        ++top;
                        continue;
                    }
                    break;
                case SortState::LeftIssued:
                {
                    auto newTop = top + 1;
                    auto [left, right] = getChildLen(top->len);

                    newTop->srcBuf = top->srcBuf;
                    newTop->destBuf = 1 - top->destBuf;
                    newTop->start = top->start + left;
                    newTop->len = right;
                    newTop->state = SortState::Init;
                    top->state = SortState::RightIssued;
                    ++top;
                    continue;
                }
                break;
                case SortState::RightIssued:
                {
                    auto [_left, _right] = getChildLen(top->len);
                    op = CommandOp::Merge;
                    srcBuf = 1 - top->destBuf;
                    destBuf = top -> destBuf;
                    start = top -> start;
                    left = _left;
                    right = _right;
                    // mergeSort<method, T>(buf[1 - top->destBuf] + top->start, left,
                    //                      buf[1 - top->destBuf] + top->start + left, right,
                    //                      buf[top->destBuf] + top->start);
                    --top;
                    return;
                }
                break;
                default:
                    break;
                }
            }
        }
    };

    template<Size32 T>
    void safeCopy(const T* fromBegin,const T* fromEnd, T* to){
        using namespace SpSpInst;
        using VecRegT = VReg<T>;
        size_t len = fromEnd - fromBegin;
        size_t p=0;
        while(p < len){
            VecPred pred = whilelt(p,len);
            VecRegT v = load(pred,fromBegin+p);
            store(pred,to+p,v);
            p+=cpu.v;
        }
    }
}; // namespace SortAlg

// https://stackoverflow.com/questions/32672474/how-to-replace-top-element-of-heap-efficiently-withouth-re-establishing-heap-inv

#pragma once
#include <functional> // less
#include <iterator> // iterator_traits
#include <utility> // move

template <typename DifferenceT>
DifferenceT heap_parent(DifferenceT k)
{
    return (k - 1) / 2;
}

template <typename DifferenceT>
DifferenceT heap_left(DifferenceT k)
{
    return 2 * k + 1;
}

template<typename RandomIt, typename Compare = std::less<>>
void heapreplace(RandomIt first, RandomIt last, Compare comp = Compare())
{
    auto const size = last - first;
    if (size <= 1)
        return;
    typename std::iterator_traits<RandomIt>::difference_type k = 0;
    auto e = std::move(first[k]);
    auto const max_k = heap_parent(size - 1);
    while (k <= max_k) {
        auto max_child = heap_left(k);
        if (max_child < size - 1 && comp(first[max_child], first[max_child + 1]))
            ++max_child; // Go to right sibling.
        if (!comp(e, first[max_child]))
            break;
        first[k] = std::move(first[max_child]);
        k = max_child;
    }

    first[k] = std::move(e);
}

template<typename RandomIt, typename Compare = std::less<>>
void heappush(RandomIt first, RandomIt last, Compare comp = Compare())
{
    auto k = last - first - 1; // k = last valid
    auto e = std::move(first[k]);

    while (k > 0 && comp(first[heap_parent(k)], e)) {
        first[k] = std::move(first[heap_parent(k)]);
        k = heap_parent(k);
    }

    first[k] = std::move(e);
}
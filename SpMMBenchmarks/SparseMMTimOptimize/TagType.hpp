#pragma once

template<bool T>
struct OptionalFloat{};

template<>
struct OptionalFloat<true>{
    float v;
    float operator*(float b){
        return v*b;
    }
};

template<>
struct OptionalFloat<false>{
    float operator*(float b){
        return b;
    }
};
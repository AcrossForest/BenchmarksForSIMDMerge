  #pragma once
  #include <vector>
  #include <algorithm>
  #include <string>
  #include <iostream>

  template<class T>
  bool arrayEqual(int len, const std::vector<T>&a, const std::vector<T>&b){
    return std::equal(a.begin(),a.begin()+len, b.begin());
  };

  template<class T>
  bool assertEqual(std::string comment, int len, const std::vector<T>&a, const std::vector<T>&b){
    int c = 0;
    for(int i=0; i<len; ++i){
        c += a[i] == b[i];
    }
    bool result = c == len;
    if (result){
        std::cout << "Check: " << comment << "\t Pass!" << std::endl;
    } else {
        std::cout << "Check: " << comment << "\t Fail! (" << c << '/' << len << ')' << std::endl;
    }
    return result;
  };
  template<class T>
  int countEqual(int len, const std::vector<T>&a, const std::vector<T>&b){
    int c=0;
    for(int i=0; i<len; ++i){
        c += a[i] == b[i];
    }
    return c;
  };
#pragma once
#include "SpSpInterface/SpSpEnums.hpp"
#ifdef __SPSP_USE_ARM__
#include "SpSpInterface/SVE/SpSpInstUserHeader.hpp"
using namespace SPSP;
#else
#include "SpSpInterface/FakeSVE/FakeSpSpInstUserHeader.hpp"
using namespace FakeSPSP;
#endif
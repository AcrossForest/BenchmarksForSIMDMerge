# on X86, 10000
[           Sort]       mean =   1.22e+06(ns)   sd =   3.86e+05(ns)     nsample =     5
[    Stable Sort]       mean =   1.04e+06(ns)   sd =   3.51e+03(ns)     nsample =     5


[      SpSp Sort]       mean =   5.07e+07(ns)   sd =   5.43e+05(ns)     nsample =     5

# on ARM, 1000
[           Sort]       mean =   4.93e+04(ns)   sd =   2.48e+03(ns)     nsample =     5
[    Stable Sort]       mean =   5.94e+04(ns)   sd =   7.84e+02(ns)     nsample =     5
[      SpSp Sort]       mean =   2.65e+04(ns)   sd =   2.25e+02(ns)     nsample =     5 // 4
[      SpSp Sort]       mean =   7.58e+03(ns)   sd =   1.03e+02(ns)     nsample =     5 // 16
[      SpSp Sort]       mean =   2.52e+03(ns)   sd =   8.57e+01(ns)     nsample =     5 // 64

# on ARM, 10000
[           Sort]       mean =   6.57e+05(ns)   sd =   3.24e+03(ns)     nsample =     5
[    Stable Sort]       mean =   7.88e+05(ns)   sd =   3.60e+02(ns)     nsample =     5
[      SpSp Sort]       mean =   3.46e+05(ns)   sd =   2.63e+02(ns)     nsample =     5 // 4
[      SpSp Sort]       mean =   9.29e+04(ns)   sd =   1.58e+02(ns)     nsample =     5 // 16
[      SpSp Sort]       mean =   2.87e+04(ns)   sd =   1.75e+02(ns)     nsample =     5 // 64


## What if the use optimism latency
[      SpSp Sort]       mean =   3.30e+05(ns)   sd =   4.09e+02(ns)     nsample =     5 // 4
[      SpSp Sort]       mean =   7.26e+04(ns)   sd =   1.52e+02(ns)     nsample =     5 // 16
[      SpSp Sort]       mean =   2.21e+04(ns)   sd =   1.57e+02(ns)     nsample =     5 // 64

## What if we use supper big L1d (16MB) and optimism latency
[      SpSp Sort]       mean =   3.29e+05(ns)   sd =   4.19e+02(ns)     nsample =     5 // 4
[      SpSp Sort]       mean =   7.15e+04(ns)   sd =   1.55e+02(ns)     nsample =     5 // 16
[      SpSp Sort]       mean =   1.98e+04(ns)   sd =   1.65e+02(ns)     nsample =     5 // 64

## Optimize BEPermute Pipeline
[      SpSp Sort]       mean =   3.45e+05(ns)   sd =   3.44e+02(ns)     nsample =     5 // 4
[      SpSp Sort]       mean =   9.29e+04(ns)   sd =   1.56e+02(ns)     nsample =     5 // 16
[      SpSp Sort]       mean =   2.87e+04(ns)   sd =   1.63e+02(ns)     nsample =     5 // 64
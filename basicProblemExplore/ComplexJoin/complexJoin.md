# key=(int), val=(float) or (float,float)
[     Add Scalar]       mean =   1.70e+05(ns)   sd =   5.21e+02(ns)     nsample =     3 // SIMD 4
[       Add SIMD]       mean =   1.05e+05(ns)   sd =   7.73e+01(ns)     nsample =     3 // SIMD 4
[     Mul Scalar]       mean =   1.52e+05(ns)   sd =   4.57e+02(ns)     nsample =     3 // SIMD 4
[       Mul SIMD]       mean =   6.85e+04(ns)   sd =   5.45e+01(ns)     nsample =     3 // SIMD 4
[Mul ReIm Scalar]       mean =   2.01e+05(ns)   sd =   6.30e+01(ns)     nsample =     3 // SIMD 4
[  Mul ReIm SIMD]       mean =   1.14e+05(ns)   sd =   2.94e+03(ns)     nsample =     3 // SIMD 4
[ShortPath Scalar]      mean =   2.31e+05(ns)   sd =   2.27e+02(ns)     nsample =     3 // SIMD 4
[ ShortPath SIMD]       mean =   1.27e+05(ns)   sd =   9.37e+01(ns)     nsample =     3 // SIMD 4

[       Add SIMD]       mean =   5.38e+04(ns)   sd =   8.02e+01(ns)     nsample =     3 // SIMD 8
[       Mul SIMD]       mean =   3.57e+04(ns)   sd =   6.38e+01(ns)     nsample =     3 // SIMD 8
[  Mul ReIm SIMD]       mean =   5.76e+04(ns)   sd =   6.68e+01(ns)     nsample =     3 // SIMD 8
[ ShortPath SIMD]       mean =   6.51e+04(ns)   sd =   1.33e+02(ns)     nsample =     3 // SIMD 8

[       Add SIMD]       mean =   2.78e+04(ns)   sd =   5.10e+01(ns)     nsample =     3 // SIMD 16
[       Mul SIMD]       mean =   1.90e+04(ns)   sd =   5.23e+01(ns)     nsample =     3 // SIMD 16
[  Mul ReIm SIMD]       mean =   2.97e+04(ns)   sd =   8.07e+01(ns)     nsample =     3 // SIMD 16
[ ShortPath SIMD]       mean =   3.35e+04(ns)   sd =   7.67e+01(ns)     nsample =     3 // SIMD 16

[       Add SIMD]       mean =   1.49e+04(ns)   sd =   5.35e+01(ns)     nsample =     3 // SIMD 32
[       Mul SIMD]       mean =   1.05e+04(ns)   sd =   5.86e+01(ns)     nsample =     3 // SIMD 32
[  Mul ReIm SIMD]       mean =   1.59e+04(ns)   sd =   6.58e+01(ns)     nsample =     3 // SIMD 32
[ ShortPath SIMD]       mean =   1.79e+04(ns)   sd =   8.21e+01(ns)     nsample =     3 // SIMD 32

[       Add SIMD]       mean =   1.01e+04(ns)   sd =   3.96e+01(ns)     nsample =     3 // SIMD 64
[       Mul SIMD]       mean =   7.51e+03(ns)   sd =   4.92e+01(ns)     nsample =     3 // SIMD 64
[  Mul ReIm SIMD]       mean =   1.13e+04(ns)   sd =   5.99e+01(ns)     nsample =     3 // SIMD 64
[ ShortPath SIMD]       mean =   1.22e+04(ns)   sd =   4.66e+01(ns)     nsample =     3 // SIMD 64

# key=(int,int), val=(float) or (float,float)
[     Add Scalar]       mean =   2.02e+05(ns)   sd =   2.32e+02(ns)     nsample =     3 // SIMD 4
[       Add SIMD]       mean =   1.44e+05(ns)   sd =   1.12e+02(ns)     nsample =     3 // SIMD 4
[     Mul Scalar]       mean =   1.63e+05(ns)   sd =   2.19e+02(ns)     nsample =     3 // SIMD 4
[       Mul SIMD]       mean =   8.28e+04(ns)   sd =   5.69e+01(ns)     nsample =     3 // SIMD 4
[Mul ReIm Scalar]       mean =   2.60e+05(ns)   sd =   6.26e+02(ns)     nsample =     3 // SIMD 4
[  Mul ReIm SIMD]       mean =   1.26e+05(ns)   sd =   3.02e+01(ns)     nsample =     3 // SIMD 4

[       Add SIMD]       mean =   7.56e+04(ns)   sd =   9.66e+01(ns)     nsample =     3 // SIMD 8
[       Mul SIMD]       mean =   4.17e+04(ns)   sd =   4.98e+01(ns)     nsample =     3 // SIMD 8
[  Mul ReIm SIMD]       mean =   6.11e+04(ns)   sd =   6.96e+01(ns)     nsample =     3 // SIMD 8

[       Add SIMD]       mean =   3.94e+04(ns)   sd =   1.53e+02(ns)     nsample =     3 // SIMD 16
[       Mul SIMD]       mean =   2.19e+04(ns)   sd =   5.43e+01(ns)     nsample =     3 // SIMD 16
[  Mul ReIm SIMD]       mean =   3.15e+04(ns)   sd =   6.84e+01(ns)     nsample =     3 // SIMD 16

[       Add SIMD]       mean =   2.04e+04(ns)   sd =   4.65e+02(ns)     nsample =     3 // SIMD 32
[       Mul SIMD]       mean =   1.18e+04(ns)   sd =   4.83e+01(ns)     nsample =     3 // SIMD 32
[  Mul ReIm SIMD]       mean =   1.71e+04(ns)   sd =   6.99e+01(ns)     nsample =     3 // SIMD 32

[       Add SIMD]       mean =   1.47e+04(ns)   sd =   2.03e+01(ns)     nsample =     3 // SIMD 64
[       Mul SIMD]       mean =   1.09e+04(ns)   sd =   4.53e+01(ns)     nsample =     3 // SIMD 64
[  Mul ReIm SIMD]       mean =   1.47e+04(ns)   sd =   7.45e+01(ns)     nsample =     3 // SIMD 64


# key=(int,int,int), val=(float) or (float,float)
[     Add Scalar]       mean =   2.06e+05(ns)   sd =   1.81e+02(ns)     nsample =     3 // SIMD 4
[       Add SIMD]       mean =   1.72e+05(ns)   sd =   2.05e+01(ns)     nsample =     3 // SIMD 4
[     Mul Scalar]       mean =   1.84e+05(ns)   sd =   2.51e+02(ns)     nsample =     3 // SIMD 4
[       Mul SIMD]       mean =   1.04e+05(ns)   sd =   6.06e+01(ns)     nsample =     3 // SIMD 4
[Mul ReIm Scalar]       mean =   2.30e+05(ns)   sd =   7.03e+01(ns)     nsample =     3 // SIMD 4
[  Mul ReIm SIMD]       mean =   1.43e+05(ns)   sd =   4.88e+01(ns)     nsample =     3 // SIMD 4

[       Add SIMD]       mean =   8.84e+04(ns)   sd =   8.66e+01(ns)     nsample =     3 // SIMD 8
[       Mul SIMD]       mean =   5.17e+04(ns)   sd =   6.64e+01(ns)     nsample =     3 // SIMD 8
[  Mul ReIm SIMD]       mean =   6.94e+04(ns)   sd =   5.53e+01(ns)     nsample =     3 // SIMD 8

[       Add SIMD]       mean =   4.60e+04(ns)   sd =   6.41e+01(ns)     nsample =     3 // SIMD 16
[       Mul SIMD]       mean =   2.70e+04(ns)   sd =   7.18e+01(ns)     nsample =     3 // SIMD 16
[  Mul ReIm SIMD]       mean =   3.61e+04(ns)   sd =   8.49e+01(ns)     nsample =     3 // SIMD 16

[       Add SIMD]       mean =   2.42e+04(ns)   sd =   5.85e+01(ns)     nsample =     3 // SIMD 32
[       Mul SIMD]       mean =   1.46e+04(ns)   sd =   5.90e+01(ns)     nsample =     3 // SIMD 32
[  Mul ReIm SIMD]       mean =   1.95e+04(ns)   sd =   8.06e+01(ns)     nsample =     3 // SIMD 32

[       Add SIMD]       mean =   2.03e+04(ns)   sd =   4.33e+01(ns)     nsample =     3 // SIMD 64
[       Mul SIMD]       mean =   1.42e+04(ns)   sd =   7.36e+01(ns)     nsample =     3 // SIMD 64
[  Mul ReIm SIMD]       mean =   1.84e+04(ns)   sd =   8.35e+01(ns)     nsample =     3 // SIMD 64


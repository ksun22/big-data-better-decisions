Ridge/Lasso Regression
================
2022-06-13

Insurance Data

Ridge Regression

``` r
# Write your code here
#load package for ridge/lasso commands
library(glmnet)
```

    ## Loading required package: Matrix

    ## Loaded glmnet 4.1-4

``` r
library(stats)

Insurance_Data <- read.csv("AutoInsurance.csv")

# Standardize the data
# Split data into test and training sets

# Set outcome variable
outcome = Insurance_Data$Purchase
outcome <- ifelse(outcome=="Yes", 1, 0)

#use model.matrix to define predictors where salary is the outcome.  -1 omits the intercept
#this function also changes qualitative variables into dummy variables
dum=model.matrix(Purchase~.,Insurance_Data)[,-1]
head(dum)
```

    ##   MOSTYPE MAANTHUI MGEMOMV MGEMLEEF MOSHOOFD MGODRK MGODPR MGODOV MGODGE MRELGE
    ## 1      33        1       3        2        8      0      5      1      3      7
    ## 2      37        1       2        2        8      1      4      1      4      6
    ## 3      37        1       2        2        8      0      4      2      4      3
    ## 4       9        1       3        3        3      2      3      2      4      5
    ## 5      40        1       4        2       10      1      4      1      4      7
    ## 6      23        1       2        1        5      0      5      0      5      0
    ##   MRELSA MRELOV MFALLEEN MFGEKIND MFWEKIND MOPLHOOG MOPLMIDD MOPLLAAG MBERHOOG
    ## 1      0      2        1        2        6        1        2        7        1
    ## 2      2      2        0        4        5        0        5        4        0
    ## 3      2      4        4        4        2        0        5        4        0
    ## 4      2      2        2        3        4        3        4        2        4
    ## 5      1      2        2        4        4        5        4        0        0
    ## 6      6      3        3        5        2        0        5        4        2
    ##   MBERZELF MBERBOER MBERMIDD MBERARBG MBERARBO MSKA MSKB1 MSKB2 MSKC MSKD
    ## 1        0        1        2        5        2    1     1     2    6    1
    ## 2        0        0        5        0        4    0     2     3    5    0
    ## 3        0        0        7        0        2    0     5     0    4    0
    ## 4        0        0        3        1        2    3     2     1    4    0
    ## 5        5        4        0        0        0    9     0     0    0    0
    ## 6        0        0        4        2        2    2     2     2    4    2
    ##   MHHUUR MHKOOP MAUT1 MAUT2 MAUT0 MZFONDS MZPART MINKM30 MINK3045 MINK4575
    ## 1      1      8     8     0     1       8      1       0        4        5
    ## 2      2      7     7     1     2       6      3       2        0        5
    ## 3      7      2     7     0     2       9      0       4        5        0
    ## 4      5      4     9     0     0       7      2       1        5        3
    ## 5      4      5     6     2     1       5      4       0        0        9
    ## 6      9      0     5     3     3       9      0       5        2        3
    ##   MINK7512 MINK123M MINKGEM MKOOPKLA PWAPART PWABEDR PWALAND PPERSAUT PBESAUT
    ## 1        0        0       4        3       0       0       0        6       0
    ## 2        2        0       5        4       2       0       0        0       0
    ## 3        0        0       3        4       2       0       0        6       0
    ## 4        0        0       4        4       0       0       0        6       0
    ## 5        0        0       6        3       0       0       0        0       0
    ## 6        0        0       3        3       0       0       0        6       0
    ##   PMOTSCO PVRAAUT PAANHANG PTRACTOR PWERKT PBROM PLEVEN PPERSONG PGEZONG
    ## 1       0       0        0        0      0     0      0        0       0
    ## 2       0       0        0        0      0     0      0        0       0
    ## 3       0       0        0        0      0     0      0        0       0
    ## 4       0       0        0        0      0     0      0        0       0
    ## 5       0       0        0        0      0     0      0        0       0
    ## 6       0       0        0        0      0     0      0        0       0
    ##   PWAOREG PBRAND PZEILPL PPLEZIER PFIETS PINBOED PBYSTAND AWAPART AWABEDR
    ## 1       0      5       0        0      0       0        0       0       0
    ## 2       0      2       0        0      0       0        0       2       0
    ## 3       0      2       0        0      0       0        0       1       0
    ## 4       0      2       0        0      0       0        0       0       0
    ## 5       0      6       0        0      0       0        0       0       0
    ## 6       0      0       0        0      0       0        0       0       0
    ##   AWALAND APERSAUT ABESAUT AMOTSCO AVRAAUT AAANHANG ATRACTOR AWERKT ABROM
    ## 1       0        1       0       0       0        0        0      0     0
    ## 2       0        0       0       0       0        0        0      0     0
    ## 3       0        1       0       0       0        0        0      0     0
    ## 4       0        1       0       0       0        0        0      0     0
    ## 5       0        0       0       0       0        0        0      0     0
    ## 6       0        1       0       0       0        0        0      0     0
    ##   ALEVEN APERSONG AGEZONG AWAOREG ABRAND AZEILPL APLEZIER AFIETS AINBOED
    ## 1      0        0       0       0      1       0        0      0       0
    ## 2      0        0       0       0      1       0        0      0       0
    ## 3      0        0       0       0      1       0        0      0       0
    ## 4      0        0       0       0      1       0        0      0       0
    ## 5      0        0       0       0      1       0        0      0       0
    ## 6      0        0       0       0      0       0        0      0       0
    ##   ABYSTAND
    ## 1        0
    ## 2        0
    ## 3        0
    ## 4        0
    ## 5        0
    ## 6        0

``` r
dum = scale(dum)
outcome = scale(outcome)

set.seed(1) # Do not change this line

#assign training sample
train=sample(1:nrow(dum), nrow(dum)/2)

#assign leftover to test sample
test=(-train)

#set testing outcome
outcome.test=outcome[test]

#Fit ridge regression, family='binomial'
#make grid of lambda values
grid=10^seq(10,-2,length=100)

#run ridge regressions for different lambda
#note alpha=0 for ridge
ridge.mod=glmnet(dum[train,],outcome[train],alpha=0,family="binomial", lambda=grid)

#display intercept and first 2 predictor vars at different lambdas
t(coef(ridge.mod)[1:3,])
```

    ## 100 x 3 sparse Matrix of class "dgCMatrix"
    ##     (Intercept)       MOSTYPE      MAANTHUI
    ## s0    -2.891823 -1.605467e-12 -2.257315e-13
    ## s1    -2.891823 -2.122333e-12 -2.984038e-13
    ## s2    -2.891823 -2.805599e-12 -3.944722e-13
    ## s3    -2.891823 -3.708837e-12 -5.214691e-13
    ## s4    -2.891823 -4.902864e-12 -6.893514e-13
    ## s5    -2.891823 -6.481297e-12 -9.112820e-13
    ## s6    -2.891823 -8.567893e-12 -1.204661e-12
    ## s7    -2.891823 -1.132625e-11 -1.592491e-12
    ## s8    -2.891823 -1.497264e-11 -2.105180e-12
    ## s9    -2.891823 -1.979295e-11 -2.782924e-12
    ## s10   -2.891823 -2.616511e-11 -3.678861e-12
    ## s11   -2.891823 -3.458873e-11 -4.863238e-12
    ## s12   -2.891823 -4.572427e-11 -6.428915e-12
    ## s13   -2.891823 -6.044479e-11 -8.498647e-12
    ## s14   -2.891823 -7.990446e-11 -1.123471e-11
    ## s15   -2.891823 -1.056290e-10 -1.485163e-11
    ## s16   -2.891823 -1.396353e-10 -1.963298e-11
    ## s17   -2.891823 -1.845897e-10 -2.595364e-11
    ## s18   -2.891823 -2.440167e-10 -3.430918e-11
    ## s19   -2.891823 -3.225757e-10 -4.535472e-11
    ## s20   -2.891823 -4.264261e-10 -5.995627e-11
    ## s21   -2.891823 -5.637102e-10 -7.925866e-11
    ## s22   -2.891823 -7.451917e-10 -1.047753e-10
    ## s23   -2.891823 -9.850995e-10 -1.385068e-10
    ## s24   -2.891823 -1.302244e-09 -1.830978e-10
    ## s25   -2.891823 -1.721489e-09 -2.420445e-10
    ## s26   -2.891823 -2.275708e-09 -3.199686e-10
    ## s27   -2.891823 -3.008351e-09 -4.229797e-10
    ## s28   -2.891823 -3.976863e-09 -5.591542e-10
    ## s29   -2.891823 -5.257179e-09 -7.391690e-10
    ## s30   -2.891823 -6.949681e-09 -9.771379e-10
    ## s31   -2.891823 -9.187069e-09 -1.291719e-09
    ## s32   -2.891823 -1.214476e-08 -1.707576e-09
    ## s33   -2.891823 -1.605466e-08 -2.257315e-09
    ## s34   -2.891823 -2.122332e-08 -2.984038e-09
    ## s35   -2.891823 -2.805597e-08 -3.944723e-09
    ## s36   -2.891823 -3.708833e-08 -5.214692e-09
    ## s37   -2.891823 -4.902858e-08 -6.893516e-09
    ## s38   -2.891823 -6.481287e-08 -9.112822e-09
    ## s39   -2.891823 -8.567876e-08 -1.204662e-08
    ## s40   -2.891823 -1.132622e-07 -1.592492e-08
    ## s41   -2.891823 -1.497259e-07 -2.105181e-08
    ## s42   -2.891823 -1.979285e-07 -2.782926e-08
    ## s43   -2.891823 -2.616495e-07 -3.678865e-08
    ## s44   -2.891823 -3.458846e-07 -4.863244e-08
    ## s45   -2.891823 -4.572379e-07 -6.428925e-08
    ## s46   -2.891823 -6.044395e-07 -8.498664e-08
    ## s47   -2.891823 -7.990298e-07 -1.123474e-07
    ## s48   -2.891823 -1.056264e-06 -1.485168e-07
    ## s49   -2.891823 -1.396308e-06 -1.963307e-07
    ## s50   -2.891823 -1.845818e-06 -2.595380e-07
    ## s51   -2.891823 -2.440029e-06 -3.430947e-07
    ## s52   -2.891823 -3.225516e-06 -4.535522e-07
    ## s53   -2.891822 -4.263839e-06 -5.995715e-07
    ## s54   -2.891822 -5.636365e-06 -7.926019e-07
    ## s55   -2.891822 -7.450629e-06 -1.047780e-06
    ## s56   -2.891821 -9.848745e-06 -1.385114e-06
    ## s57   -2.891821 -1.301850e-05 -1.831059e-06
    ## s58   -2.891820 -1.720802e-05 -2.420587e-06
    ## s59   -2.891819 -2.274507e-05 -3.199935e-06
    ## s60   -2.891818 -3.006253e-05 -4.230231e-06
    ## s61   -2.891816 -3.973198e-05 -5.592301e-06
    ## s62   -2.891814 -5.250774e-05 -7.393016e-06
    ## s63   -2.891811 -6.938491e-05 -9.773694e-06
    ## s64   -2.891808 -9.167520e-05 -1.292123e-05
    ## s65   -2.891803 -1.211062e-04 -1.708282e-05
    ## s66   -2.891797 -1.599502e-04 -2.258546e-05
    ## s67   -2.891791 -2.111917e-04 -2.986185e-05
    ## s68   -2.891783 -2.787415e-04 -3.948464e-05
    ## s69   -2.891775 -3.677101e-04 -5.221206e-05
    ## s70   -2.891767 -4.847501e-04 -6.904844e-05
    ## s71   -2.891763 -6.384771e-04 -9.132492e-05
    ## s72   -2.891767 -8.399719e-04 -1.208070e-04
    ## s73   -2.891790 -1.103353e-03 -1.598382e-04
    ## s74   -2.891866 -1.432873e-03 -2.120700e-04
    ## s75   -2.892001 -1.867922e-03 -2.809428e-04
    ## s76   -2.892261 -2.424503e-03 -3.723798e-04
    ## s77   -2.892731 -3.129245e-03 -4.938715e-04
    ## s78   -2.893547 -4.009634e-03 -6.554166e-04
    ## s79   -2.894914 -5.090345e-03 -8.703301e-04
    ## s80   -2.897122 -6.387266e-03 -1.156259e-03
    ## s81   -2.900558 -7.899125e-03 -1.536410e-03
    ## s82   -2.905695 -9.597620e-03 -2.040973e-03
    ## s83   -2.913072 -1.142500e-02 -2.708553e-03
    ## s84   -2.923181 -1.326680e-02 -3.588302e-03
    ## s85   -2.936430 -1.498206e-02 -4.740869e-03
    ## s86   -2.953031 -1.639742e-02 -6.239732e-03
    ## s87   -2.972965 -1.732658e-02 -8.170656e-03
    ## s88   -2.996003 -1.758533e-02 -1.062875e-02
    ## s89   -3.021781 -1.700098e-02 -1.371217e-02
    ## s90   -3.049873 -1.542567e-02 -1.751209e-02
    ## s91   -3.079839 -1.273680e-02 -2.209846e-02
    ## s92   -3.111242 -8.841840e-03 -2.750239e-02
    ## s93   -3.143659 -3.683215e-03 -3.369612e-02
    ## s94   -3.176701  2.762339e-03 -4.057675e-02
    ## s95   -3.210019  1.048901e-02 -4.795902e-02
    ## s96   -3.243303  1.942733e-02 -5.557858e-02
    ## s97   -3.276285  2.951282e-02 -6.313417e-02
    ## s98   -3.308775  4.071849e-02 -7.032392e-02
    ## s99   -3.340582  5.293630e-02 -7.687835e-02

``` r
# Select optimal lambda values through cross validation

set.seed(1) 

#cross validation to select lambda
cv.out=cv.glmnet(dum[train ,],outcome[train],family="binomial",alpha =0)

#plot CV error for different lambda
plot(cv.out)
```

![](Ridge-Lasso-Regression_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

``` r
#select optimal lambda with smallest CV error
#Note this lambda is approximately equal to 0
bestlam=cv.out$lambda.min
bestlam
```

    ## [1] 0.05600354

``` r
#fit ridge regression using the optimal lambda
#find the mean squared error rate

#run model with optimal lambda on test data
ridge.pred=predict(ridge.mod,s=bestlam,newx=dum[test ,])

#calculate MSE for comparison with lasso
mean((ridge.pred-outcome.test)^2)
```

    ## [1] 11.34498

``` r
#run ridge regression with optimal lambda on full dataset
out=glmnet(dum,outcome,alpha=0,lambda=grid)
ridge.coef=predict(out,type="coefficients",s=bestlam )

#display coefficients
ridge.coef = drop(ridge.coef)
ridge.coef
```

    ##   (Intercept)       MOSTYPE      MAANTHUI       MGEMOMV      MGEMLEEF 
    ##  3.823146e-16  1.688966e-02 -1.050658e-02 -6.379717e-03  3.153661e-02 
    ##      MOSHOOFD        MGODRK        MGODPR        MGODOV        MGODGE 
    ## -1.462757e-02 -1.604639e-02  9.356518e-03  8.506922e-03 -8.613467e-03 
    ##        MRELGE        MRELSA        MRELOV      MFALLEEN      MFGEKIND 
    ##  2.706031e-02 -5.573502e-03  7.489494e-03 -7.868889e-03 -1.558323e-02 
    ##      MFWEKIND      MOPLHOOG      MOPLMIDD      MOPLLAAG      MBERHOOG 
    ##  3.498225e-03  4.295907e-02  1.749556e-03 -4.959012e-02  1.013154e-02 
    ##      MBERZELF      MBERBOER      MBERMIDD      MBERARBG      MBERARBO 
    ##  1.346329e-03 -2.608194e-02  2.723432e-02 -4.710387e-03  8.306277e-03 
    ##          MSKA         MSKB1         MSKB2          MSKC          MSKD 
    ## -6.067361e-04 -7.954456e-03 -2.768012e-03  2.140308e-02 -1.566995e-03 
    ##        MHHUUR        MHKOOP         MAUT1         MAUT2         MAUT0 
    ## -1.931165e-02  8.242976e-03  2.613481e-02  1.611373e-02  6.196596e-03 
    ##       MZFONDS        MZPART       MINKM30      MINK3045      MINK4575 
    ##  4.404325e-03 -1.790656e-02  1.739522e-02  1.841705e-02  8.077925e-03 
    ##      MINK7512      MINK123M       MINKGEM      MKOOPKLA       PWAPART 
    ##  1.899865e-02 -2.813788e-02  2.581538e-02  2.147756e-02  4.522048e-02 
    ##       PWABEDR       PWALAND      PPERSAUT       PBESAUT       PMOTSCO 
    ## -9.967866e-03 -1.687019e-02  9.866380e-02  2.254216e-04 -1.521464e-02 
    ##       PVRAAUT      PAANHANG      PTRACTOR        PWERKT         PBROM 
    ## -1.197978e-02  2.176336e-02  1.563314e-02 -5.064677e-03  7.863834e-03 
    ##        PLEVEN      PPERSONG       PGEZONG       PWAOREG        PBRAND 
    ## -4.244602e-02  4.100816e-03  4.740790e-02  5.699923e-02  7.734110e-02 
    ##       PZEILPL      PPLEZIER        PFIETS       PINBOED      PBYSTAND 
    ## -2.301293e-02 -3.574393e-03  4.831260e-03 -2.803166e-02  1.846532e-03 
    ##       AWAPART       AWABEDR       AWALAND      APERSAUT       ABESAUT 
    ## -5.887695e-03  6.075459e-03 -1.215522e-02  2.882499e-02 -7.891998e-03 
    ##       AMOTSCO       AVRAAUT      AAANHANG      ATRACTOR        AWERKT 
    ##  1.213436e-02  1.590387e-03 -1.267855e-02 -2.515452e-02 -2.710840e-03 
    ##         ABROM        ALEVEN      APERSONG       AGEZONG       AWAOREG 
    ## -8.171619e-03  4.258178e-02 -9.294048e-03 -2.934747e-02 -2.921860e-02 
    ##        ABRAND       AZEILPL      APLEZIER        AFIETS       AINBOED 
    ## -2.742591e-02  3.363470e-02  9.387065e-02  2.507396e-02  2.568177e-02 
    ##      ABYSTAND 
    ##  3.349234e-02

``` r
#all non-zero coefficients #
#ridge regression does not perform variable selection
#no coefficient out of 86 total equals to 0
```

Lasso Regression

``` r
# Fit lasso regression to the same data
set.seed(1) 

#make grid of lambda values
grid=10^seq(10,-2,length=100)

#run lasso regressions for different lambda
#note alpha=1 for lasso
lasso.mod=glmnet(dum[train,],outcome[train],family="binomial",alpha=1,lambda=grid)

#display intercept and first 2 predictor vars at different lambdas
t(coef(lasso.mod)[1:3,])
```

    ## 100 x 3 sparse Matrix of class "dgCMatrix"
    ##     (Intercept) MOSTYPE MAANTHUI
    ## s0    -2.891823       .        .
    ## s1    -2.891823       .        .
    ## s2    -2.891823       .        .
    ## s3    -2.891823       .        .
    ## s4    -2.891823       .        .
    ## s5    -2.891823       .        .
    ## s6    -2.891823       .        .
    ## s7    -2.891823       .        .
    ## s8    -2.891823       .        .
    ## s9    -2.891823       .        .
    ## s10   -2.891823       .        .
    ## s11   -2.891823       .        .
    ## s12   -2.891823       .        .
    ## s13   -2.891823       .        .
    ## s14   -2.891823       .        .
    ## s15   -2.891823       .        .
    ## s16   -2.891823       .        .
    ## s17   -2.891823       .        .
    ## s18   -2.891823       .        .
    ## s19   -2.891823       .        .
    ## s20   -2.891823       .        .
    ## s21   -2.891823       .        .
    ## s22   -2.891823       .        .
    ## s23   -2.891823       .        .
    ## s24   -2.891823       .        .
    ## s25   -2.891823       .        .
    ## s26   -2.891823       .        .
    ## s27   -2.891823       .        .
    ## s28   -2.891823       .        .
    ## s29   -2.891823       .        .
    ## s30   -2.891823       .        .
    ## s31   -2.891823       .        .
    ## s32   -2.891823       .        .
    ## s33   -2.891823       .        .
    ## s34   -2.891823       .        .
    ## s35   -2.891823       .        .
    ## s36   -2.891823       .        .
    ## s37   -2.891823       .        .
    ## s38   -2.891823       .        .
    ## s39   -2.891823       .        .
    ## s40   -2.891823       .        .
    ## s41   -2.891823       .        .
    ## s42   -2.891823       .        .
    ## s43   -2.891823       .        .
    ## s44   -2.891823       .        .
    ## s45   -2.891823       .        .
    ## s46   -2.891823       .        .
    ## s47   -2.891823       .        .
    ## s48   -2.891823       .        .
    ## s49   -2.891823       .        .
    ## s50   -2.891823       .        .
    ## s51   -2.891823       .        .
    ## s52   -2.891823       .        .
    ## s53   -2.891823       .        .
    ## s54   -2.891823       .        .
    ## s55   -2.891823       .        .
    ## s56   -2.891823       .        .
    ## s57   -2.891823       .        .
    ## s58   -2.891823       .        .
    ## s59   -2.891823       .        .
    ## s60   -2.891823       .        .
    ## s61   -2.891823       .        .
    ## s62   -2.891823       .        .
    ## s63   -2.891823       .        .
    ## s64   -2.891823       .        .
    ## s65   -2.891823       .        .
    ## s66   -2.891823       .        .
    ## s67   -2.891823       .        .
    ## s68   -2.891823       .        .
    ## s69   -2.891823       .        .
    ## s70   -2.891823       .        .
    ## s71   -2.891823       .        .
    ## s72   -2.891823       .        .
    ## s73   -2.891823       .        .
    ## s74   -2.891823       .        .
    ## s75   -2.891823       .        .
    ## s76   -2.891823       .        .
    ## s77   -2.891823       .        .
    ## s78   -2.891823       .        .
    ## s79   -2.891823       .        .
    ## s80   -2.891823       .        .
    ## s81   -2.891823       .        .
    ## s82   -2.891823       .        .
    ## s83   -2.891823       .        .
    ## s84   -2.891823       .        .
    ## s85   -2.891823       .        .
    ## s86   -2.891823       .        .
    ## s87   -2.891823       .        .
    ## s88   -2.891823       .        .
    ## s89   -2.891823       .        .
    ## s90   -2.891823       .        .
    ## s91   -2.891823       .        .
    ## s92   -2.891823       .        .
    ## s93   -2.891823       .        .
    ## s94   -2.891823       .        .
    ## s95   -2.892022       .        .
    ## s96   -2.905968       .        .
    ## s97   -2.942618       .        .
    ## s98   -2.992620       .        .
    ## s99   -3.048119       .        .

``` r
# Select optimal lambda through cross-validation

set.seed(1) # Do not change this line
#cross validation to select lambda
cv.out=cv.glmnet(dum[train ,],outcome[train],family="binomial",alpha =1)

#plot CV error for different lambda
plot(cv.out)
```

![](Ridge-Lasso-Regression_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
#select optimal lambda with smallest CV error
#note lambda is approx 0
bestlam=cv.out$lambda.min
bestlam
```

    ## [1] 0.005345809

``` r
# fit lasso regression using optimal lambda 
# report the mean squared error
#run model with optimal lambda on test data
lasso.pred=predict(lasso.mod,s=bestlam,newx=dum[test ,])

#calculate mean squared error for comparison with ridge
mean((lasso.pred-outcome.test)^2)
```

    ## [1] 10.66017

``` r
mean((ridge.pred-outcome.test)^2)
```

    ## [1] 11.34498

``` r
#run lasso model with optimal lambda on full dataset
out=glmnet(dum,outcome,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam )
#display coefficients
lasso.coef = drop(lasso.coef)
lasso.coef
```

    ##   (Intercept)       MOSTYPE      MAANTHUI       MGEMOMV      MGEMLEEF 
    ##  3.815164e-16  0.000000e+00  0.000000e+00  0.000000e+00  1.062009e-02 
    ##      MOSHOOFD        MGODRK        MGODPR        MGODOV        MGODGE 
    ##  0.000000e+00 -7.362792e-03  4.570966e-03  0.000000e+00 -7.582829e-03 
    ##        MRELGE        MRELSA        MRELOV      MFALLEEN      MFGEKIND 
    ##  2.010600e-02 -5.745773e-03  0.000000e+00  0.000000e+00  0.000000e+00 
    ##      MFWEKIND      MOPLHOOG      MOPLMIDD      MOPLLAAG      MBERHOOG 
    ##  0.000000e+00  2.750870e-02  0.000000e+00 -2.739852e-02  0.000000e+00 
    ##      MBERZELF      MBERBOER      MBERMIDD      MBERARBG      MBERARBO 
    ##  0.000000e+00 -2.664991e-02  1.279356e-02  0.000000e+00  0.000000e+00 
    ##          MSKA         MSKB1         MSKB2          MSKC          MSKD 
    ##  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00 
    ##        MHHUUR        MHKOOP         MAUT1         MAUT2         MAUT0 
    ## -1.793831e-02  0.000000e+00  1.260887e-02  0.000000e+00  0.000000e+00 
    ##       MZFONDS        MZPART       MINKM30      MINK3045      MINK4575 
    ##  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00 
    ##      MINK7512      MINK123M       MINKGEM      MKOOPKLA       PWAPART 
    ##  7.901393e-03 -1.851955e-02  1.857114e-02  2.075346e-02  3.335217e-02 
    ##       PWABEDR       PWALAND      PPERSAUT       PBESAUT       PMOTSCO 
    ##  0.000000e+00 -1.877815e-02  1.180460e-01  0.000000e+00  0.000000e+00 
    ##       PVRAAUT      PAANHANG      PTRACTOR        PWERKT         PBROM 
    ## -8.066223e-04  0.000000e+00  0.000000e+00 -3.315006e-03  0.000000e+00 
    ##        PLEVEN      PPERSONG       PGEZONG       PWAOREG        PBRAND 
    ##  0.000000e+00  0.000000e+00  1.270311e-02  2.141804e-02  5.033805e-02 
    ##       PZEILPL      PPLEZIER        PFIETS       PINBOED      PBYSTAND 
    ##  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00 
    ##       AWAPART       AWABEDR       AWALAND      APERSAUT       ABESAUT 
    ##  0.000000e+00  0.000000e+00  0.000000e+00  6.707737e-03 -1.488407e-04 
    ##       AMOTSCO       AVRAAUT      AAANHANG      ATRACTOR        AWERKT 
    ##  0.000000e+00  0.000000e+00  0.000000e+00 -8.075117e-03  0.000000e+00 
    ##         ABROM        ALEVEN      APERSONG       AGEZONG       AWAOREG 
    ##  0.000000e+00  1.130989e-03  0.000000e+00  0.000000e+00  0.000000e+00 
    ##        ABRAND       AZEILPL      APLEZIER        AFIETS       AINBOED 
    ##  0.000000e+00  7.059340e-03  8.820269e-02  2.344438e-02  0.000000e+00 
    ##      ABYSTAND 
    ##  2.987584e-02

``` r
#~32 non-zero, ~54 zero coefficients estimated (out of 86)
#expected as lasso performs variable selection, setting equal to 0
#ridge does not this
```

*Conclusion:*

The mean squared error for lasso is smaller (lasso fits data better than
ridge). Given that it also has many more non-zero coefficients with the
optimal lambda through variable selection, it is probably preferable.
This is probably because the data is “sparse” with only a few
significant parameters, such that the rest can be set to (exactly) 0
through variable selection.

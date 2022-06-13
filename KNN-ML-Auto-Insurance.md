KNN ML Auto Insurance
================
Kevin Sun
5/3/2022

*KNN Model with Auto Insurance Data*

Clear working memory

``` r
rm(list=ls())
```

Fit a KNN model to the data from the Caravan auto insurance dataset
using outcome Purchase and all of the other variables in the dataset as
predictors.

``` r
# Write your code here
library(class) # package for knn classifier
Insurance_Data <- read.csv("AutoInsurance.csv")
attach(Insurance_Data)
#quick summary of data
colnames(Insurance_Data)
```

    ##  [1] "MOSTYPE"  "MAANTHUI" "MGEMOMV"  "MGEMLEEF" "MOSHOOFD" "MGODRK"  
    ##  [7] "MGODPR"   "MGODOV"   "MGODGE"   "MRELGE"   "MRELSA"   "MRELOV"  
    ## [13] "MFALLEEN" "MFGEKIND" "MFWEKIND" "MOPLHOOG" "MOPLMIDD" "MOPLLAAG"
    ## [19] "MBERHOOG" "MBERZELF" "MBERBOER" "MBERMIDD" "MBERARBG" "MBERARBO"
    ## [25] "MSKA"     "MSKB1"    "MSKB2"    "MSKC"     "MSKD"     "MHHUUR"  
    ## [31] "MHKOOP"   "MAUT1"    "MAUT2"    "MAUT0"    "MZFONDS"  "MZPART"  
    ## [37] "MINKM30"  "MINK3045" "MINK4575" "MINK7512" "MINK123M" "MINKGEM" 
    ## [43] "MKOOPKLA" "PWAPART"  "PWABEDR"  "PWALAND"  "PPERSAUT" "PBESAUT" 
    ## [49] "PMOTSCO"  "PVRAAUT"  "PAANHANG" "PTRACTOR" "PWERKT"   "PBROM"   
    ## [55] "PLEVEN"   "PPERSONG" "PGEZONG"  "PWAOREG"  "PBRAND"   "PZEILPL" 
    ## [61] "PPLEZIER" "PFIETS"   "PINBOED"  "PBYSTAND" "AWAPART"  "AWABEDR" 
    ## [67] "AWALAND"  "APERSAUT" "ABESAUT"  "AMOTSCO"  "AVRAAUT"  "AAANHANG"
    ## [73] "ATRACTOR" "AWERKT"   "ABROM"    "ALEVEN"   "APERSONG" "AGEZONG" 
    ## [79] "AWAOREG"  "ABRAND"   "AZEILPL"  "APLEZIER" "AFIETS"   "AINBOED" 
    ## [85] "ABYSTAND" "Purchase"

``` r
# Data standardization and training for knn
standardized.X=scale(Insurance_Data[,-86])
test=1:1000

train.X=standardized.X[-test ,] # exclude test sample
test.X=standardized.X[test ,]

train.Y=Purchase[-test] # exclude test sample
test.Y=Purchase[test]
```

KNN model with K=2 with confusion matrix. Hypothetical success rate and
profits

``` r
#Set seed
set.seed(1) 
#Generate KNN model with K=2
nearest1_pred=knn(train = train.X, test= test.X, cl= train.Y,k=2)
# Confusion Matrix: Rate of Predicted vs Actual Values
table(nearest1_pred,test.Y)
```

    ##              test.Y
    ## nearest1_pred  No Yes
    ##           No  883  51
    ##           Yes  58   8

``` r
#Success Rate
8/(8+58)
```

    ## [1] 0.1212121

``` r
# random guess positive rate
(51+8)/1000
```

    ## [1] 0.059

``` r
# Profits with marketing strategy suggested by KNN
-66*(180)+8*(1800)
```

    ## [1] 2520

``` r
# End of your code
```

KNN model with K=5 with confusion matrix. Hypothetical success rate and
profits

``` r
set.seed(1) # Set seed
#Generate KNN model with K=5
nearest1_pred=knn(train = train.X, test= test.X, cl= train.Y,k=5)
# Confusion Matrix: Rate of Predicted vs Actual Values
table(nearest1_pred,test.Y)
```

    ##              test.Y
    ## nearest1_pred  No Yes
    ##           No  930  55
    ##           Yes  11   4

``` r
#Success Rate
4/(11+4)
```

    ## [1] 0.2666667

``` r
# random guess positive rate
(55+4)/1000
```

    ## [1] 0.059

``` r
# Profits with marketing strategy suggested by KNN
-15*(180)+4*(1800)
```

    ## [1] 4500

``` r
# End of your code
```

KNN model with K=10 with confusion matrix. Hypothetical success rates
and profits.

``` r
set.seed(1) # Set seed
#Generate KNN model with K=10
nearest1_pred=knn(train = train.X, test= test.X, cl= train.Y,k=10)
# Confusion Matrix: Rate of Predicted vs Actual Values
table(nearest1_pred,test.Y)
```

    ##              test.Y
    ## nearest1_pred  No Yes
    ##           No  941  58
    ##           Yes   0   1

``` r
#Success Rate
1/(0+1)
```

    ## [1] 1

``` r
# random guess positive rate
(58+1)/1000
```

    ## [1] 0.059

``` r
# Profits with marketing strategy suggested by KNN
-1*(180)+1*(1800)
```

    ## [1] 1620

``` r
# End of your code
```

*Conclusion:* As k in the KNN increases, the success rate increases.
This makes sense as increasing amount of nearest neighbors decreases
variance, but also increases bias. There is a bias-variance tradeoff.
However, profits do not necessarily increase and go up with k. Profits
are higher when k=5 than k=2 or k=10 as the success rate is more optimal
in both chance and quantity. Hence, it is best to balance the variance
and bias, selecting an optimal amount of nearest neighbors that will
maximize profit (e.g. k=5).

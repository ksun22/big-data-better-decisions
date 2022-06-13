Random Forest Trees
================
5/16/2022

``` r
# Tree regression and random forest on auto insurance data
# Load required packages
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(ggplot2)

# For the tree functions
library(rpart)
library(party)
```

    ## Loading required package: grid

    ## Loading required package: mvtnorm

    ## Loading required package: modeltools

    ## Loading required package: stats4

    ## Loading required package: strucchange

    ## Loading required package: zoo

    ## 
    ## Attaching package: 'zoo'

    ## The following objects are masked from 'package:base':
    ## 
    ##     as.Date, as.Date.numeric

    ## Loading required package: sandwich

``` r
# For Random Forest
library(randomForest)
```

    ## randomForest 4.7-1

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

``` r
# For Boosting
library(gbm)
```

    ## Loaded gbm 2.1.8

``` r
Auto <- read.csv("AutoInsurance.csv")

# create train and test sets (~40% for training, ~60% for testing)
nrow(Auto)
```

    ## [1] 5822

``` r
smp_size <- floor(0.4 * nrow(Auto))
smp_size
```

    ## [1] 2328

``` r
## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(1:nrow(Auto), size = smp_size)

Auto.train <- Auto[train_ind, ]
Auto.test <- Auto[-train_ind, ]

# Run random forest on training data
#Display importance factors
# Set seed
set.seed(1234)

#Fit the training data
# nodesize is minimum size of terminal nodes
# mtry is number of variables randomly sampled as candidates at each split
# ntree is the number of bootstrap samples
model.rf <- randomForest(as.factor(Purchase) ~ ., 
                         data = Auto.train, mtry = 2, nodesize = 30, ntree = 500)
#Use model to predict test dataset outcomes
#Compare predicted vs. actual outcomes in confusion matrix
#Calculate the error rate

#predict outcomes on the test data
pred.rf <- predict(model.rf, newdata = Auto.test)
#Code below should be 1?
pred.rf<-ifelse(pred.rf=="1", 1, 0)
#compare actual and predicted outcomes
y <- Auto.test$Purchase
y_test <- ifelse(y=="Yes", 1, 0)
conf.rf <- table(pred.rf, y_test)
conf.rf
```

    ##        y_test
    ## pred.rf    0    1
    ##       0 3292  202

``` r
# Evaluate using MSE
#calculate the error rate
y_pred <- pred.rf
mean((y_pred-y_test)^2)
```

    ## [1] 0.05781339

``` r
(conf.rf[1,2])/sum(conf.rf)
```

    ## [1] 0.05781339

``` r
# End of your code
```

*Conclusion:* I notice that I get a 1 x 2 matrix. This likely means the
test sample has no positive observable data, i.e. it is an imbalanced
dataset. There are 3292+202 negative observables with an error rate of
202/(3292+202)=0.05781.

``` r
# Create another Random Forest changing mtry & nodesize
pred.rf <- predict(model.rf, newdata = Auto.train)
pred.rf<-ifelse(pred.rf=="1", 1, 0)
#compare actual and predicted outcomes
#change y to numerical variable
y <- Auto.train$Purchase
y_train <- ifelse(y=="Yes", 1, 0)
conf.rf <- table(pred.rf, y_train)
conf.rf
```

    ##        y_train
    ## pred.rf    0    1
    ##       0 2182  146

``` r
# Evaluate using MSE (error rate)
y_pred <- pred.rf
mean((y_pred-y_train)^2)
```

    ## [1] 0.06271478

``` r
#compare to error rate of previous model
202/(3292+202)
```

    ## [1] 0.05781339

``` r
#Slightly different error rates
```

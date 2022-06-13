# Write your code here
#load package for ridge/lasso commands
library(glmnet)
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
# Select optimal lambda values through cross validation

set.seed(1) 

#cross validation to select lambda
cv.out=cv.glmnet(dum[train ,],outcome[train],family="binomial",alpha =0)

#plot CV error for different lambda
plot(cv.out)

#select optimal lambda with smallest CV error
#Note this lambda is approximately equal to 0
bestlam=cv.out$lambda.min
bestlam

#fit ridge regression using the optimal lambda
#find the mean squared error rate

#run model with optimal lambda on test data
ridge.pred=predict(ridge.mod,s=bestlam,newx=dum[test ,])

#calculate MSE for comparison with lasso
mean((ridge.pred-outcome.test)^2)

#run ridge regression with optimal lambda on full dataset
out=glmnet(dum,outcome,alpha=0,lambda=grid)
ridge.coef=predict(out,type="coefficients",s=bestlam )

#display coefficients
ridge.coef = drop(ridge.coef)
ridge.coef
#all non-zero coefficients #
#ridge regression does not perform variable selection
#no coefficient out of 86 total equals to 0
# Fit lasso regression to the same data
set.seed(1) 

#make grid of lambda values
grid=10^seq(10,-2,length=100)

#run lasso regressions for different lambda
#note alpha=1 for lasso
lasso.mod=glmnet(dum[train,],outcome[train],family="binomial",alpha=1,lambda=grid)

#display intercept and first 2 predictor vars at different lambdas
t(coef(lasso.mod)[1:3,])

# Select optimal lambda through cross-validation

set.seed(1) # Do not change this line
#cross validation to select lambda
cv.out=cv.glmnet(dum[train ,],outcome[train],family="binomial",alpha =1)

#plot CV error for different lambda
plot(cv.out)

#select optimal lambda with smallest CV error
#note lambda is approx 0
bestlam=cv.out$lambda.min
bestlam

# fit lasso regression using optimal lambda 
# report the mean squared error
#run model with optimal lambda on test data
lasso.pred=predict(lasso.mod,s=bestlam,newx=dum[test ,])

#calculate mean squared error for comparison with ridge
mean((lasso.pred-outcome.test)^2)
mean((ridge.pred-outcome.test)^2)
#run lasso model with optimal lambda on full dataset
out=glmnet(dum,outcome,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam )
#display coefficients
lasso.coef = drop(lasso.coef)
lasso.coef
#~32 non-zero, ~54 zero coefficients estimated (out of 86)
#expected as lasso performs variable selection, setting equal to 0
#ridge does not this
```

*Conclusion:*

The mean squared error for lasso is smaller (lasso fits data better than ridge). Given that it also has many more non-zero coefficients with the optimal lambda through variable selection, it is probably preferable. This is probably because the data is "sparse" with only a few significant parameters, such that the rest can be set to (exactly) 0 through variable selection. 

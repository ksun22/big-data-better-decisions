#K-Nearest Neighbors Machine Learning Auto Insurance

library(class) # package for knn classifier
Insurance_Data <- read.csv("AutoInsurance.csv")
attach(Insurance_Data)
#quick summary of data
colnames(Insurance_Data)
# Data standardization and training for knn
standardized.X=scale(Insurance_Data[,-86])
test=1:1000

train.X=standardized.X[-test ,] # exclude test sample
test.X=standardized.X[test ,]

train.Y=Purchase[-test] # exclude test sample
test.Y=Purchase[test]

#Set seed
set.seed(1) 
#Generate KNN model with K=2
nearest1_pred=knn(train = train.X, test= test.X, cl= train.Y,k=2)
# Confusion Matrix: Rate of Predicted vs Actual Values
table(nearest1_pred,test.Y)
#Success Rate
8/(8+58)
# random guess positive rate
(51+8)/1000
# Profits with marketing strategy suggested by KNN
-66*(180)+8*(1800)

set.seed(1) # Set seed
#Generate KNN model with K=5
nearest1_pred=knn(train = train.X, test= test.X, cl= train.Y,k=5)
# Confusion Matrix: Rate of Predicted vs Actual Values
table(nearest1_pred,test.Y)
#Success Rate
4/(11+4)
# random guess positive rate
(55+4)/1000
# Profits with marketing strategy suggested by KNN
-15*(180)+4*(1800)

set.seed(1) # Set seed
#Generate KNN model with K=10
nearest1_pred=knn(train = train.X, test= test.X, cl= train.Y,k=10)
# Confusion Matrix: Rate of Predicted vs Actual Values
table(nearest1_pred,test.Y)
#Success Rate
1/(0+1)
# random guess positive rate
(58+1)/1000
# Profits with marketing strategy suggested by KNN
-1*(180)+1*(1800)

#Conclusion: As k in the KNN increases, the success rate increases. 
#This makes sense as increasing amount of nearest neighbors decreases variance, but also increases bias. 
#There is a bias-variance tradeoff. 
#However, profits do not necessarily increase and go up with k. 
#Profits are higher when k=5 than k=2 or k=10 as the success rate is more optimal in both chance and quantity. 
#Hence, it is best to balance the variance and bias, selecting an optimal amount of nearest neighbors that will maximize profit (e.g. k=5).
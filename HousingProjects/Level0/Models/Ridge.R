library(Metrics)
library(usdm)
library(lars)
library(boot)
setwd("../..")
source('plotting.R')

# Set seed for reproducability for random desitions
set.seed(1)

# data <- read.csv('DataSets/train.csv', header=TRUE)
Training <- read.csv('ForestFilledData/train.csv', header=TRUE) # precomputed replacement of missing values using missForest


Num_NA<-sapply(Training,function(y)length(which(is.na(y)==T)))
NA_Count<- data.frame(Item=colnames(Training),Count=Num_NA)

Training<- Training[,-c(7,73,74,75)]
Num<-sapply(Training,is.numeric)
Num<-Training[,Num]

for(i in 1:77){
  if(is.factor(Training[,i])){
    Training[,i]<-as.integer(Training[,i])
  }
}


Training[is.na(Training)]<-0
Num[is.na(Num)]<-0

train=sample(1460, 196)
lars.X = Training[,-1] # Remove Id
lars.X = lars.X[,-76] # Remove SalePrice

# Ridge Regression Model
library(glmnet)
y=as.matrix(Training$SalePrice)
x=as.matrix(lars.X)
grid=10^seq(10,-2,length=100)
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)
train=sample(1:nrow(x),nrow(x)/2)
test=(-train)
y.test=y[test]

cv.out=cv.glmnet(x[train,],y[train], alpha=0)
plot(cv.out)
bestlam=cv.out$lambda.min
ridge.pred=predict(ridge.mod,s=bestlam,newx=x[test,])
err=mean((ridge.pred-y.test)^2)
cat("Best Lambda: ", bestlam)
cat("\n")
cat("Best Lambda: ", bestlam)
cat("\n")
cat("Validation MSE: ", err)
cat("\n")
cat("Validation RMSE: ", rmse(log(ridge.pred), log(y.test)))
cat("\n")

# Model for full data
Test = read.csv('ForestFilledData/test.csv', header=TRUE)
Test<- Test[,-c(7,73,74,75)]
Num_NA<-sapply(Test,function(y)length(which(is.na(y)==T)))
NA_Count<- data.frame(Item=colnames(Test),Count=Num_NA)

Num<-sapply(Test,is.numeric)
Num<-Test[,Num]

for(i in 1:76){
  if(is.factor(Test[,i])){
    Test[,i]<-as.integer(Test[,i])
  }
}
lars.test.X = Test[,-1] # Remove Id
sum(is.na(lars.test.X))
cv.out=cv.glmnet(x, y, alpha=0)
bestlam=cv.out$lambda.min
ridge.mod=glmnet(x,y,alpha=0,lambda=bestlam)
ridge.pred=predict(ridge.mod,s=bestlam,newx=as.matrix(lars.test.X))

cat("Best Lambda on Test: ", bestlam)
cat("\n")
pred = data.frame(matrix(unlist(ridge.pred)), nrow=1459, byrow=T)
colnames(pred) <- c("SalePrice")
results <- cbind(Test$Id, pred)
# Write results
results = results[,1:2]
dim(results)
fix(results)

write.csv(results, file = "Level0/Predictions/ridge.csv")


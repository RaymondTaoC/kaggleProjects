library(DMwR)
library(randomForest)
library(missForest)
library(doParallel)
library(Metrics)

registerDoParallel(cores=4)

# data <- read.csv('DataSets/train.csv', header=TRUE)
setwd("../..")
Housing <- read.csv('ForestFilledData/train.csv', header=TRUE) # precomputed replacement of missing values using missForest

for(i in 1:81){
  if(is.factor(Housing[,i])){
    Housing[,i]<-as.integer(Housing[,i])
  }
}

# Pop Ids column
Ids <- Housing[,1]
Housing <- Housing[,-1]

# Test with a validation set
# train = sample(1:nrow(Housing), nrow(Housing)/2)
#housing.test = Housing[-train,"SalePrice"]

# Test Model
#rf.housing = randomForest(SalePrice~.,data=Housing,subset=train,mtry=27,ntree=1000)
#yhat.rf=predict(rf.housing,newdata=Housing[-train,])
#plot(yhat.rf,housing.test)
#abline(0,1)
#print("Final:")
#mean((yhat.rf-housing.test)^2)
#rmse(log(yhat.rf),log(housing.test))


test_data <- read.csv('ForestFilledData/test.csv', header=TRUE) # precomputed replacement of missing values using missForest

for(i in 1:80){
  if(is.factor(test_data[,i])){
    data[,i]<-as.integer(test_data[,i])
  }
}

Ids <- test_data[,2]
test_data <- test_data[,-c(1, 2)]

colnames(test_data)
# Full model
rf.housing = randomForest(SalePrice~.,data=Housing,mtry=27,ntree=10000)
yhat.rf=predict(rf.housing,newdata=test_data)
results <- cbind(Ids, yhat.rf)
# Write results
dim(results)
fix(results)

write.csv(results, file = "Level0/Predictions/rf.csv")


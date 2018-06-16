library(pls)

set.seed(123)

# data <- read.csv('DataSets/train.csv', header=TRUE)
setwd("../..")
data <- read.csv('ForestFilledData/train.csv', header=TRUE) # precomputed replacement of missing values using missForest

set.seed(123)

for(i in 1:81){
  if(is.factor(data[,i])){
    data[,i]<-as.integer(data[,i])
  }
}

# Pop Ids column
data <- data[,-c(1,2)]
Housing <- data

pcr.fit=pcr(SalePrice~., data=Housing, scale=FALSE, validation='CV')
validationplot(pcr.fit, val.type='RMSEP')

pcr_final.fit=pcr(SalePrice~., data=Housing, scale=FALSE, ncomp=54)
summary(pcr_final.fit)

test_data <- read.csv('ForestFilledData/test.csv', header=TRUE) # precomputed replacement of missing values using missForest

for(i in 1:80){
  if(is.factor(test_data[,i])){
    data[,i]<-as.integer(test_data[,i])
  }
}

# Pop Ids column
Ids <- test_data[,2]
test_data <- test_data[,-c(1,2)]

yhat.pcr=predict(pcr_final.fit,newdata=test_data,ncomp=54)
results <- cbind(Ids, yhat.pcr)
# Write results
dim(results)
fix(results)

write.csv(results, file = "Level0/Predictions/pcr.csv")




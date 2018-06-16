library(nnet)
library(doParallel)
library(Metrics)


registerDoParallel(cores =3)

# data <- read.csv('DataSets/train.csv', header=TRUE)
data <- read.csv('mfHousing.csv', header=TRUE) # precomputed replacement of missing values using missForest

set.seed(123)

for(i in 1:81){
  if(is.factor(data[,i])){
    data[,i]<-as.integer(data[,i])
  }
}

# Pop Ids column
Ids <- data[,1]
data <- data[,-c(1,2,3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18,19,20,21, 22,23,24,25,26)]
Housing <- data

# Test with a validation set
train = sample(1:nrow(Housing), nrow(Housing)/2)
housing.test = Housing[-train,"SalePrice"]

colnames(Housing)
nnet.fit <- nnet(SalePrice~., data=Housing[train,], size=20, linout=TRUE, MaxNWts=100000, trace=FALSE, maxit=1000000)

yhat.gbm=predict(nnet.fit)
plot(yhat.gbm,housing.test)
abline(0,1)
print("lam:0.001, ntree=5000, MSE:")
mean((yhat.gbm-housing.test)^2)
rmse(log(yhat.gbm),log(housing.test))




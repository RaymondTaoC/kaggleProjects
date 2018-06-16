library(xgboost)
library(Matrix)
library(methods)

set.seed(123)

# load data
# data <- read.csv('DataSets/train.csv', header=TRUE)
setwd("../..")
train <- read.csv('ForestFilledData/train.csv', header=TRUE) # precomputed replacement of missing values using missForest
test <- read.csv('ForestFilledData/test.csv', header=TRUE) # precomputed replacement of missing values using missForest

for(i in 1:80){
  if(is.factor(train[,i])){
    train[,i]<-as.integer(train[,i])
  }
}

Ids <- test[,2]
train <- train[,-c(1, 2)]

for(i in 1:80){
  if(is.factor(test[,i])){
    test[,i]<-as.integer(test[,i])
  }
}

test <- test[,-c(1, 2)]

re_train<- as.matrix(train,rownames.force =NA)
re_train<- as(re_train,'sparseMatrix')
retrain_Data<- xgb.DMatrix(data = re_train[,-80],label=re_train[,"SalePrice"])

param<-list(
  objective = "reg:linear",
  eval_metric = "rmse",
  booster = "gbtree",
  max_depth = 8,
  eta = 0.123,
  gamma = 0.0385, 
  subsample = 0.734,
  colsample_bytree = 0.512
)

bstSparse_retrain<- xgb.train(params=param,
                              data=retrain_Data,
                              nrounds = 600,
                              watchlist = list(train = retrain_Data),
                              verbose = TRUE,
                              print_every_n = 50,
                              nthread = 2
                              )

Test_Matrix<-as.matrix(test,rownames.force = FALSE)
Test_Matrix<-as(Test_Matrix,"sparseMatrix")
Test_Matrix<-xgb.DMatrix(data = as.matrix(test))

#predict model
pred<- predict(bstSparse_retrain, newdata=Test_Matrix)
Submit<-cbind(Id= Ids, SalePrice= pred)
# Write results
dim(Submit)
fix(Submit)

write.csv(Submit, file = "Level0/Predictions/xgb.csv")


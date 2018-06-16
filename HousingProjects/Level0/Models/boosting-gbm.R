library(DMwR)
library(gbm)
library(doParallel)
library(Metrics)
library(caret)
library(ggplot2)


cl <- makePSOCKcluster(5)
registerDoParallel(cl)

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
Ids <- data[,1]
data <- data[,-1]
Housing <- data



# Training validation Model
gbm.housing = gbm(SalePrice~., data=Housing, distribution="gaussian", n.tree=5000, interaction.depth=4, shrinkage = 0.1)

#summary(gbm.housing)
#yhat.gbm=predict(gbm.housing,newdata=Housing, n.tree=5000, interaction.depth=4)
#plot(yhat.gbm,housing.test)
#abline(0,1)
#print("lam:0.001, ntree=5000, MSE:")
#mean((yhat.gbm-housing.test)^2)
#rmse(log(yhat.gbm),log(housing.test))
#stop()
test_data <- read.csv('ForestFilledData/test.csv', header=TRUE) # precomputed replacement of missing values using missForest

for(i in 1:80){
  if(is.factor(test_data[,i])){
    data[,i]<-as.integer(test_data[,i])
  }
}

# Pop Ids column
Ids <- test_data[,1]
data <- test_data[,-1]

# Full model
gbm.housing = gbm(SalePrice~., data=Housing, distribution="gaussian", n.tree=5000,interaction.depth=4,shrinkage = 0.1)
yhat.gbm=predict(gbm.housing,newdata=test_data, n.tree=5000, interaction.depth=4)

results <- cbind(test_data$Id, yhat.gbm)
# Write results
dim(results)
fix(results)

write.csv(results, file = "Level0/Predictions/gbm.csv")

## Param Search
#fitControl <- trainControl(## 10-fold CV
#                           method = "repeatedcv",
#                           number = 10,
#                           ## repeated ten times
#                           repeats = 10)
#gbmGrid <-  expand.grid(interaction.depth = 1:5, 
#                        n.trees = (1:5)*1000, 
#                        shrinkage = 0.1,
#			n.minobsinnode = 20)
#gbmFit2 <- train(SalePrice~.,data=Housing, 
#                 method = "gbm", 
#                 trControl = fitControl, 
#                 verbose = FALSE, 
#                 ## Now specify the exact models 
#                 ## to evaluate:
#                 tuneGrid = gbmGrid)
#gbmFit2
#ggplot(gbmFit2)



stopCluster(cl)


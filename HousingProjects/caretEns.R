library(caret)
library(caretEnsemble)
library(doParallel)

registerDoParallel(cores=4)

###### load and clean training data ######
data <- read.csv('ForestFilledData/train.csv', header=TRUE) # precomputed replacement of missing values using missForest

set.seed(123)

for(i in 1:81){
  if(is.factor(data[,i])){
    data[,i]<-as.integer(data[,i])
  }
}

# Pop Ids column
data <- data[,-c(1,2)]
dataset <- data

start.time <- Sys.time()
print(start.time)
###### create submodels ######
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('pcr', 'blassoAveraged', 'rf','gbm', 'xgbLinear')
models <- caretList(SalePrice~., 
data=dataset, 
trControl=control, 
methodList=algorithmList)
save(models, file='caretModels.rda')
results <- resamples(models)

# correlation between results
modelCor(results)
splom(results)

###### Load test data ######
test_data <- read.csv('ForestFilledData/test.csv', header=TRUE) # precomputed replacement of missing values using missForest

for(i in 1:80){
  if(is.factor(test_data[,i])){
    data[,i]<-as.integer(test_data[,i])
  }
}

# Pop Id column
Id <- test_data[,2]
Pred <- test_data[,-c(1,2)]

###### train Stack and predict ######
# stack using gbm
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
stack.gbm <- caretStack(models, method="gbm", metric="RMSE", trControl=stackControl)

SalePrice = predict(stack.gbm, Pred)
gbm_results = cbind(Id, SalePrice)
write.csv(gbm_results, file = "caretPred/gbm.csv", row.names=FALSE)


# stack using random forest
stack.rf <- caretStack(models, method="rf", metric="RMSE", trControl=stackControl)
print(stack.rf)

rf_stack.pred = predict(stack.rf, Pred)
SalePrice = cbind(Id, SalePrice)
write.csv(rf_results, file = "caretPred/rf.csv", row.names=FALSE)

# stack using Deep Neural Network 
stack.brnn <- caretStack(models, method="pcr", metric="RMSE", trControl=stackControl)
print(stack.brnn)

brnn_stack.pred = predict(stack.brnn, Pred)
SalePrice = cbind(Id, SalePrice)
write.csv(brnn_results, file = "caretPred/pcr.csv", row.names=FALSE)

print('Final Time')
end.time <- Sys.time()
time.taken <- end.time - start.time
print('Total Time')
print(time.taken)
summary(results)
dotplot(results)


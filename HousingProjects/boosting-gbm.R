# Require dependencies
install.packages("DMwR")
install.packages("gbm")
install.packages("missForest")
install.packages("doParallel")
install.packages("Metrics")
install.packages("caret")
install.packages("ggplot2")

library(DMwR)
library(gbm)
library(missForest)
library(doParallel)
library(Metrics)
library(caret)
library(ggplot2)


cl <- makePSOCKcluster(5)
registerDoParallel(cl)

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
data <- data[,-1]
write.csv(cbind(Ids, data), file = "test.csv")

# Replace NANs
housing.missForest <- missForest(
xmis=data,
mtry=27,
ntree=1000,
variablewise = FALSE,
decreasing = FALSE,
verbose = FALSE,
replace = TRUE,
classwt = NULL,
cutoff = NULL,
strata = NULL,
sampsize = NULL,
nodesize = NULL,
maxnodes = NULL,
xtrue = NA,
parallelize = "variables"
)
Housing <- housing.missForest$ximp
write.csv(cbind(Ids, Housing), file = "mfHousing.csv")

# Test with a validation set
train = sample(1:nrow(Housing), nrow(Housing)/2)
housing.test = Housing[-train,"SalePrice"]

# Find optimal parameters with caret
fitControl <- trainControl(## 10-fold CV
                           method = "repeatedcv",
                           number = 10,
                           ## repeated ten times
                           repeats = 10)
gbmGrid <-  expand.grid(interaction.depth = 1:5, 
                        n.trees = (1:5)*1000, 
                        shrinkage = 0.1,
			n.minobsinnode = 20)
gbmFit2 <- train(SalePrice~.,data=Housing, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid)
gbmFit2
ggplot(gbmFit2)

# Final Model
gbm.housing = gbm(SalePrice~.,
data=Housing[train,],
distribution="gaussian",
n.tree=5000,
interaction.depth=4,
shrinkage = 0.1)

summary(gbm.housing)
yhat.gbm=predict(gbm.housing,newdata=Housing[-train,], n.tree=5000, interaction.depth=4)
plot(yhat.gbm,housing.test)
abline(0,1)
print("lam:0.001, ntree=5000, MSE:")
mean((yhat.gbm-housing.test)^2)
rmse(log(yhat.gbm),log(housing.test))

stopCluster(cl)


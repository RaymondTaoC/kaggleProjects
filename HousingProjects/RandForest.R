library(DMwR)
library(randomForest)
library(missForest)
library(doParallel)
library(Metrics)

registerDoParallel(cores=4)

data <- read.csv('DataSets/train.csv', header=TRUE)
# data <- read.csv('mfHousing.csv', header=TRUE) # precomputed replacement of missing values using missForest

set.seed(123)

for(i in 1:81){
  if(is.factor(data[,i])){
    data[,i]<-as.integer(data[,i])
  }
}

# Pop Ids
Ids <- data[,1]
data <- data[,-1]
write.csv(cbind(Ids, data), file = "test.csv")
colnames(data)

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

# Final Model
rf.housing = randomForest(SalePrice~.,data=Housing,subset=train,mtry=27,ntree=1000)
yhat.rf=predict(rf.housing,newdata=Housing[-train,])
plot(yhat.rf,housing.test)
abline(0,1)
print("Final:")
mean((yhat.rf-housing.test)^2)
rmse(log(yhat.rf),log(housing.test))


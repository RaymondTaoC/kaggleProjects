library(DMwR)
library(randomForest)
library(missForest)
library(doParallel)
library(Metrics)

registerDoParallel(cores=4)

#### Fill Training Data ####
data <- read.csv('DataSets/train.csv', header=TRUE)

set.seed(123)

for(i in 1:81){
  if(is.factor(data[,i])){
    data[,i]<-as.integer(data[,i])
  }
}

# Pop Ids
Ids <- data[,1]
data <- data[,-1]
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
write.csv(cbind(Ids, Housing), file = "ForestFilledData/train.csv")


#### Fill Test Data ####
data <- read.csv('DataSets/test.csv', header=TRUE)
colnames(data)
ncol(data)
stop()
for(i in 1:80){
  if(is.factor(data[,i])){
    data[,i]<-as.integer(data[,i])
  }
}

# Pop Ids
Ids <- data[,1]
data <- data[,-1]
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
write.csv(cbind(Ids, Housing), file = "ForestFilledData/test.csv")


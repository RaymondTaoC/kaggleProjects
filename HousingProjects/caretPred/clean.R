setwd('../')
getwd()
test_data <- read.csv('ForestFilledData/test.csv', header=TRUE)
data <- read.csv('caretPred/rf.csv', header=TRUE)
Id <- test_data[,2]

dnn_results = cbind(Id, data$rf_stack.pred)
write.csv(dnn_results, file = "caretPred/rf0.csv", row.names=FALSE)

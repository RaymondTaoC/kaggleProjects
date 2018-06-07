library(ggplot2)
library(lars)
library(usdm)
source('plotting.R')

# Set seed for reproducability for random desitions
set.seed(123)

# Clean Data
Training <- read.csv('DataSets/train.csv', header=TRUE)

Num_NA<-sapply(Training,function(y)length(which(is.na(y)==T)))
NA_Count<- data.frame(Item=colnames(Training),Count=Num_NA)

Training<- Training[,-c(7,73,74,75)]
Num<-sapply(Training,is.numeric)
Num<-Training[,Num]

for(i in 1:77){
  if(is.factor(Training[,i])){
    Training[,i]<-as.integer(Training[,i])
  }
}


Training[is.na(Training)]<-0
Num[is.na(Num)]<-0

# Data Plot(s)
pairs(~SalePrice+OverallQual+TotalBsmtSF+GarageCars+GarageArea,data=Training,
      main="Scatterplot Matrix")
p<- ggplot(Training,aes(x= YearBuilt,y=SalePrice))+geom_point()+geom_smooth()
p

train=sample(1460, 196)
lars.X = Training[,-1] # Remove Id
lars.X = lars.X[,-76] # Remove SalePrice

# Least-Angle Regression Model
lar.fit = lars(as.matrix(lars.X),  as.matrix(Training$SalePrice), type="lar")
plot(lar.fit)
summary(lar.fit)
cv.lars(as.matrix(lars.X),  as.matrix(Training$SalePrice), type="lar", index=1:75) # pretty bad, about 1.5e9
lar.pred=predict.lars(lar.fit,lars.X, type="fit", s=20, mode="step")
print(mean((Training$SalePrice-as.numeric(unlist(lar.pred[4])))[-train]^2)) # about 1.2e9: the validation set approach confirms 



# Lasso Regression Model
lasso.fit = lars(as.matrix(lars.X),  as.matrix(Training$SalePrice), type="lasso")
plot(lasso.fit)
summary(lasso.fit)
cv.lars(as.matrix(lars.X),  as.matrix(Training$SalePrice), type="lasso") # pretty bad, about 1.5e9
lass.pred=predict.lars(lasso.fit,lars.X, type="fit", s=20, mode="step")
print(mean((Training$SalePrice-as.numeric(unlist(lass.pred[4])))[-train]^2)) # about 1.2e9: the validation set approach confirms 

# Lasso Regression Model
lasso.fit = lars(as.matrix(lars.X),  as.matrix(Training$SalePrice), type="lasso")
plot(lasso.fit)
summary(lasso.fit)
cv.lars(as.matrix(lars.X),  as.matrix(Training$SalePrice), type="lasso") # pretty bad, about 1.5e9

# Ridge Regression Model
library(glmnet)
y=as.matrix(Training$SalePrice)
x=as.matrix(lars.X)
grid=10^seq(10,-2,length=100)
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)
train=sample(1:nrow(x),nrow(x)/2)
set.seed(1)
test=(-train)

cv.out=cv.glmnet(x[train,],y[train], alpha=0)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam
stop()

# Fitting initial least squares model
reg1<- lm(SalePrice~., data = Training)
summary(reg1)

attach(Training)

# Analysis of Full model
resisduals = rstandard(reg1)

for(i in 1:ncol(Training)) {
	plot_lowess(Training[,i], resisduals)
}

detach(Training)




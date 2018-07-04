rm(list=ls(all=TRUE))


library(ggplot2)
library(caret)
library(randomForest)
library(rpart)


training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))

dim(testing)
dim(training)

table(training$classe)

#preProcess
#Feature Selection
# First remove near zero variance features, empty columns and NA columns

nzvcol1 <- nearZeroVar(training)
training <- training[,-nzvcol1]
training <- training[,-(1:7)]
training[training == ""] <- NA
training <- training[, colSums(is.na(training)) == 0] 

dim(training)

nzvcol2 <- nearZeroVar(testing)
testing <- testing[,-nzvcol2]
testing[testing == ""] <- NA
testing <- testing[,-(1:7)]
testing <- testing[, colSums(is.na(testing)) == 0] 

set.seed(23333)
intrain <- createDataPartition(training$classe,p=0.7,list=FALSE)
train <- training[intrain,]
validation <- training[-intrain,]

#Model

# random forest

modfit <- randomForest(classe~.,data=train)

print(modfit)

pred1 <- predict(modfit,validation,type="class")
confusionMatrix(validation$classe, pred1)



finalpred <- predict(modfit,testing,type="class")
finalpred

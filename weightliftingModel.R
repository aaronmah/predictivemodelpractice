# Loading data
library(caret)
library(ggplot2)
trainURL <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
download.file(trainURL, destfile="./trainURL.csv")
train <- read.csv('./trainURL.csv')

testURL <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
download.file(testURL, destfile="./testURL.csv")
testing <- read.csv('./testURL.csv')

inTrain <- createDataPartition(train$classe, p=0.75, list=FALSE)
training <- train[inTrain,]
validation <- train[-inTrain,]
dim(training); dim(validation); dim(testing)

# Exploratory
install.packages("RANN")
library(RANN)
library(dplyr)
head(training)
summary(training)
str(training)
colSums(is.na(training))

#Cleaning up data
numeric <- names(training)[!names(training) %in% c("X","user_name", "cvtd_timestamp", "new_window","classe")]
training[,numeric] <- lapply(training[,numeric], function(x) as.numeric(as.character(x)))
validation[,numeric] <- lapply(validation[,numeric], function(x) as.numeric(as.character(x)))

numeric <- names(testing)[!names(testing) %in% c("X","user_name", "cvtd_timestamp", "new_window","problem_id")]
testing[,numeric] <- lapply(testing[,numeric], function(x) as.numeric(as.character(x)))


colfactor <- names(training)[names(training) %in% c("user_name", "new_window", "classe")]
training[,colfactor] <- lapply(training[,colfactor], as.factor)
validation[,colfactor] <- lapply(validation[,colfactor], as.factor)

colfactortest <- names(testing)[names(testing) %in% c("user_name", "new_window")]
testing[,colfactortest] <- lapply(testing[,colfactortest], as.factor)

#high dimensionality, check for zero variance; a lot of columns flagged for nzv
nzv <- nearZeroVar(training,saveMetrics = TRUE)
nzv

#remove zero variance columns
training <- training[nzv$zeroVar==FALSE]
validation <- validation[,names(training)]
testing <- testing[!nzv$zeroVar==TRUE]

# Given that there's exactly the same number of NA values across these columns, I'm interpreting the NA's
# as 0, ie. sensor is in resting state. for example, 0 roll or 0 pitch when there are no values (instead of NA)

training[is.na(training)] <- 0
validation[is.na(validation)] <- 0
testing[is.na(testing)] <- 0

#attempt pca, but it complains about near zero variance columns to reduce dimensionality
preProc <- preProcess(training[,c(-1,-151)], method="pca", threshold=0.95)
trainingPCA <- predict(preProc, training)

validationPCA <- predict(preProc,validation) 
testingPCA <- predict(preProc,testing)

summary(trainingPCA)
dim(trainingPCA) # reduced to 64 predictors
str(testingPCA)

# As we're tackling a classification problem, we'll try models such as random forest, boosting (lots of potentially weak predictors) 
# and regularized reression as this may have a more linear in nature

# We'll also attempt some simple blending to evaluate effectiveness
install.packages("doParallel")
library(parallel)
library(doParallel)

## Random Forest model, at 97.67% 
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl <- trainControl(method="cv", number=5, allowParallel = TRUE)
modelrf <- train(classe~., data=trainingPCA[,-1], method='rf', trControl=fitControl)
modelrf$finalModel
modelrf
stopCluster(cluster)
registerDoSEQ()

# Testing against validation data set to test out of sample error
predict(modelrf, validationPCA)

confusionMatrix(validationPCA$classe, predict(modelrf, validationPCA[,-1])) # 98.14%, one percent short with only rf


# Boost model against training data
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
fitControl <- trainControl(method="cv", number=5, allowParallel = TRUE)
modelbst <- train(classe~., data=trainingPCA[,-1], method='gbm', trControl=fitControl) # 91.17%

modelbst
stopCluster(cluster)
registerDoSEQ()

# testing against validation set
confusionMatrix(validationPCA$classe, predict(modelbst,validationPCA[,-1])) # 90.58% against validation set

confusionMatrix(predict(modelrf, validationPCA), predict(modelbst,validationPCA[,-1]))  # 91.82% accuracy between the two models

# LDA
modelLDA <- train(classe~., data=trainingPCA[,-1], method="lda")
modelLDA #75%
predict(modelLDA, validationPCA)
confusionMatrix(validationPCA$classe, predict(modelLDA, validationPCA[,-1])) #75% again validation set

#SVM
library(e1071)
modelSVM <- svm(classe~., data=trainingPCA[,-1])
confusionMatrix(validationPCA$classe, predict(modelSVM, validationPCA[,-1])) #89.97% accuracy

# four models, with rf at best accuracy (~98.14%). Will blend models to see if we can improve accuracy to ~99%

rf <-predict(modelrf, trainingPCA[,-1])
bst <- predict(modelbst,trainingPCA[,-1])
lda <- predict(modelLDA, trainingPCA[,-1])
sv <- predict(modelSVM, trainingPCA[,-1])

#blendDF <- data.frame(randomf=rf, boost=bst, svm=sv, outcomeclass=trainingPCA$classe)

blendDF <- data.frame(randomf=rf, boost=bst, ld=lda, outcomeclass=trainingPCA$classe)

cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
fitControl <- trainControl(method="cv", number=5, allowParallel = TRUE)
modelblend <- train(outcomeclass~., data=blendDF, method='rf')
stopCluster(cluster)
registerDoSEQ()

modelblend$finalModel # looks promissing at 99.99%
modelblend

# Testing against validation
validrf <- predict(modelrf, validationPCA[,-1])
validbst <- predict(modelbst, validationPCA[,-1])
validlda <- predict(modelLDA, validationPCA[,-1])
#validsvm <- predict(modelSVM, validationPCA[,-1])

validBlend <- data.frame(randomf=validrf, boost=validbst, ld=validlda, outcomeclass=validationPCA$classe)
confusionMatrix(validBlend$outcomeclass, predict(modelblend, validBlend)) # 98.14% 

# ----- Prepare testing data


testrf <- predict(modelrf, testingPCA[,c(-1,-5)])
testbst <- predict(modelbst, testingPCA[,c(-1,-5)])
testlda <- predict(modelLDA, testingPCA[,c(-1,-5)])
#testsvm <- predict(modelSVM, testingPCA[,c(-1,-5)])

testBlend <- data.frame(randomf=testrf, boost=testbst, ld=testlda)
testAnswers <- predict(modelblend,testBlend)


testAnswers

library(doParallel)
library(corrplot)
library(caret)
library(prediction)
library(dplyr)
library(sqldf)
#library(randomForest)

#C5.0, random forest, KKNN and support vector machines

# Find how many cores are on your machine
detectCores() # Result = Typically 4 to 6

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes.
cl <- makeCluster(2)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 2

#Reading the training data
#These are the data matrices that you will use to develop your models to predict
#the overall sentiment toward iPhone and Galaxy.
# iphonesentiment is the dependent variable
data <- read.csv("../smallmatrix_labeled_8d/galaxy_smallmatrix_labeled_9d.csv")
str(data)
summary(data)
head(data)
tail(data)
sum(is.null(data))

for(i in 1:3) {
  print(hist(data[ , i]))
}

hist(data["nokialumina"])

options(max.print=1000000)
corr_data <- cor(data)
corrplot(corr_data)
findCorrelation(corr_data)
str(data)
corr_data

#No hay features altamente correlacionados con la variable dependiente
#En caso que haya se hace:
# create a new data set and remove features highly correlated with the dependant
#iphoneCOR <- data
#iphoneCOR$featureToRemove <- NULL

#Examine Feature Variance
#The distribution of values within a feature is related to how
#much information that feature holds in the data set. Features
#with no variance can be said to hold little to no information.
#Features that have very little, or "near zero variance", may or
#may not have useful information. To explore feature variance we
#can use nearZeroVar() from the caret package.

#nearZeroVar() with saveMetrics = TRUE returns an object containing
#a table including: frequency ratio, percentage unique, zero
#variance and near zero variance

nzvMetrics <- nearZeroVar(data, saveMetrics = TRUE)
nzvMetrics

#Review your table. Are there features that have zero variance?
#Near zero variance? Let’s use nearZeroVar() again to create an
#index of near zero variance features. The index will allow us
#to quickly remove features.

# nearZeroVar() with saveMetrics = FALSE returns an vector
nzv <- nearZeroVar(data, saveMetrics = FALSE)
nzv

# create a new data set and remove near zero variance features
galaxyNZV <- data[,-nzv]
str(galaxyNZV)

#Recursive Feature Elimination
#RFE is a form of automated feature selection. Caret’s rfe()
#function with random forest will try every combination of
#feature subsets and return a final list of recommended features.

# Let's sample the data before using RFE
set.seed(123)
galaxySample <- data[sample(1:nrow(data), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment)
rfeResults <- rfe(galaxySample[,1:58],
                  galaxySample$galaxysentiment,
                  sizes=(1:58),
                  rfeControl=ctrl)

# Get results
rfeResults
#La variable 19 es la mas optima

# Plot results
plot(rfeResults, type=c("g", "o"))

#The resulting table and plot display each subset and its
#accuracy and kappa. An asterisk denotes the the number of
#features that is judged the most optimal from RFE.

#After identifying features for removal, create a new data set
#and add the dependant variable.
# create new data set with rfe recommended features
galaxyRFE <- data[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
galaxyRFE$galaxysentiment <- data$galaxysentiment

# review outcome
str(galaxyRFE)

#After preprocessing you may have the following data sets:

#  data (this data set retains all of the original features for "out of the box" modeling)
#iphoneCOR: no consegui nunguna feature correlacionada con la variable dependiente
#iphoneNZV
#iphoneRFE
#Converting the dependent variable in factor
data$galaxysentiment<-as.factor(data$galaxysentiment)
#iphoneCOR$iphonesentiment<-as.factor(iphoneCOR$iphonesentiment)
galaxyNZV$galaxysentiment<-as.factor(galaxyNZV$galaxysentiment)
galaxyRFE$galaxysentiment<-as.factor(galaxyRFE$galaxysentiment)
#%%%%%%%%%%Train and test  data for data
train_all <- round(nrow(data)*0.7)
test_all <- nrow(data)-train_all

training_indices_all <- sample(seq_len(nrow(data)),size =train_all)
trainSet_all <- data[training_indices_all,]
test_indices_all <- sample(seq_len(nrow(data)),size =test_all)
testSet_all <- data[test_indices_all,]
#%%%%%%%%%%#%%%%%%%%%%#%%%%%%%%%%#%%%%%%%%%%
#%%%%%%%%%%Train and test  data for galaxyNZV
train_NZV <- round(nrow(galaxyNZV)*0.7)
test_NZV <- nrow(galaxyNZV)-train_NZV

training_indices_NZV <- sample(seq_len(nrow(galaxyNZV)),size =train_NZV)
trainSet_NZV <- galaxyNZV[training_indices_NZV,]
test_indices_NZV <- sample(seq_len(nrow(galaxyNZV)),size =test_NZV)
testSet_NZV <- galaxyNZV[test_indices_NZV,]
#%%%%%%%%%%#%%%%%%%%%%#%%%%%%%%%%#%%%%%%%%%%
#%%%%%%%%%%Train and test  data for galaxyRFE
train_RFE <- round(nrow(galaxyRFE)*0.7)
test_RFE <- nrow(galaxyRFE)-train_RFE

training_indices_RFE <- sample(seq_len(nrow(galaxyRFE)),size =train_RFE)
trainSet_RFE <- galaxyRFE[training_indices_RFE,]
test_indices_RFE <- sample(seq_len(nrow(galaxyRFE)),size =test_RFE)
testSet_RFE <- galaxyRFE[test_indices_RFE,]
#%%%%%%%%%%#%%%%%%%%%%#%%%%%%%%%%#%%%%%%%%%%
#%%%%%%%%%%#Cross-Validation%%%#%%%%%%%%%%#%%%%%%%%%%
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%GRIDs%%%%%%%%%%%%%%%%%%%%%%%%%%%
#C5.0, random forest, KKNN and support vector machines
c50Grid <- expand.grid(.trials = c(1:9, (1:10)*10),
                       .model = c("tree","rules"),
                       .winnow = c(TRUE, FALSE))
#--------------------------------------------------
rfGrid <- expand.grid(mtry=c(1,2,3))
#--------------------------------------------------
svmGrid <- expand.grid(C = 2^(1:3), sigma = seq(0.25, 2, length = 8))
#--------------------------------------------------
kknnGrid <- expand.grid(kmax = c(3, 5, 7 ,9, 11), distance = c(1, 2),
                        kernel = c("rectangular", "gaussian", "cos"))
#--------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%Training with all Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%
#train_data <- trainSet_all
system.time(c5Fit1 <- train(galaxysentiment ~ ., data = trainSet_all,
                            method = "C5.0",
                            trControl = fitControl, tuneGrid = c50Grid))
c5Fit1$finalModel$tuneValue
c5Fit1
#--------------------------------------------------
system.time(rfFit1 <- train(galaxysentiment ~ ., data = trainSet_all,
                            method = "rf",
                            trControl = fitControl, tuneGrid = rfGrid))
rfFit1$finalModel$tuneValue
rfFit1
#--------------------------------------------------
system.time(svmFit1 <- train(galaxysentiment ~ ., data = trainSet_all,
                             method = "svmRadial",
                             trControl = fitControl, tuneGrid = svmGrid))
svmFit1$finalModel
svmFit1
#--------------------------------------------------
system.time(kknnFit1 <- train(galaxysentiment ~ ., data = trainSet_all,
                              method = "kknn",
                              trControl = fitControl, tuneGrid = kknnGrid))
kknnFit1$finalModel$tuneValue
kknnFit1
#--------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%Training with all RFE %%%%%%%%%%%%%%%%%%%%%%%%%%%%
#train_data <- trainSet_RFE
system.time(c5Fit2 <- train(galaxysentiment ~ ., data = trainSet_RFE,
                            method = "C5.0",
                            trControl = fitControl, tuneGrid = c50Grid))
c5Fit2$finalModel$tuneValue
c5Fit2
#--------------------------------------------------
system.time(rfFit2 <- train(galaxysentiment ~ ., data = trainSet_RFE,
                            method = "rf",
                            trControl = fitControl, tuneGrid = rfGrid))
rfFit2$finalModel$tuneValue
rfFit2
#--------------------------------------------------
system.time(svmFit2 <- train(galaxysentiment ~ ., data = trainSet_RFE,
                             method = "svmRadial",
                             trControl = fitControl, tuneGrid = svmGrid))
svmFit2$finalModel
svmFit2
#--------------------------------------------------
system.time(kknnFit2 <- train(galaxysentiment ~ ., data = trainSet_RFE,
                              method = "kknn",
                              trControl = fitControl, tuneGrid = kknnGrid))
kknnFit2$finalModel$tuneValue
kknnFit2
#--------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%%Training with all NZV %%%%%%%%%%%%%%%%%%%%%%%%%%%%
#train_data <- trainSet_NZV
system.time(c5Fit3 <- train(galaxysentiment ~ ., data = trainSet_NZV,
                            method = "C5.0",
                            trControl = fitControl, tuneGrid = c50Grid))
c5Fit3$finalModel$tuneValue
c5Fit3
#--------------------------------------------------
system.time(rfFit3 <- train(galaxysentiment ~ ., data = trainSet_NZV,
                            method = "rf",
                            trControl = fitControl, tuneGrid = rfGrid))
rfFit3$finalModel$tuneValue
rfFit3
#--------------------------------------------------
system.time(svmFit3 <- train(galaxysentiment ~ ., data = trainSet_NZV,
                             method = "svmRadial",
                             trControl = fitControl, tuneGrid = svmGrid))
svmFit3$finalModel
svmFit3
#--------------------------------------------------
system.time(kknnFit3 <- train(galaxysentiment ~ ., data = trainSet_NZV,
                              method = "kknn",
                              trControl = fitControl, tuneGrid = kknnGrid))
kknnFit3$finalModel$tuneValue
kknnFit3
#--------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%Performance differences%%%%%%%%%%%%%%%%%%%%%
print('#%%%%%%%%%%%%Results for All Data%%%%%%%%%%%%')
resamps1 <- resamples(list(C5p0 = c5Fit1,
                           RF = rfFit1, SVM = svmFit1, KKNN=kknnFit1))
resamps1
summary(resamps1)
#--------------------------------------------------
print('#%%%%%%%%%%%%Results for RFE Data%%%%%%%%%%%%')
resamps2 <- resamples(list(C5p0 = c5Fit2,
                           RF = rfFit2, SVM = svmFit2, KKNN=kknnFit2))
resamps2
summary(resamps2)

plot(c5Fit2, metric = "Kappa")
plot(rfFit2, metric = "Kappa")
plot(svmFit2, metric = "Kappa")
plot(kknnFit2, metric = "Kappa")

plot(c5Fit2, metric = "Accuracy")
plot(rfFit2, metric = "Accuracy")
plot(svmFit2, metric = "Accuracy")
plot(kknnFit2, metric = "Accuracy")

#--------------------------------------------------
print('#%%%%%%%%%%%%Results for NZV Data%%%%%%%%%%%%')
resamps3 <- resamples(list(C5p0 = c5Fit3,
                           RF = rfFit3, SVM = svmFit3, KKNN=kknnFit3))
resamps3
summary(resamps3)
print('#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#--------------------------------------------------
#%%%%%%%%%%%%%%%%%%%%%Predictions with All Data%%%%%%%%%%%%%%%%%%%%%
#testSet <- testSet_all
str(testSet_all)
predictions_c51 <- prediction(c5Fit1, testSet_all, se.fit = FALSE)
str(predictions_c51)
plot(predictions_c51$galaxysentiment,testSet_all$galaxysentiment)
cmC5 <- confusionMatrix(predictions_c51$fitted,testSet_all$galaxysentiment)
cmC5
#--------------------------------------------------
predictions_rf1 <- prediction(rfFit1, testSet_all, se.fit = FALSE)
str(predictions_rf1)
plot(predictions_rf1$galaxysentiment,testSet_all$galaxysentiment)
cmRF <- confusionMatrix(predictions_rf1$fitted,testSet_all$galaxysentiment)
cmRF
#--------------------------------------------------
predictions_svm1 <- prediction(svmFit1, testSet_all, se.fit = FALSE)
str(predictions_svm1)
plot(predictions_svm1$galaxysentiment,testSet_all$galaxysentiment)
cmSVM <- confusionMatrix(predictions_svm1$fitted,testSet_all$galaxysentiment)
cmSVM
#--------------------------------------------------
predictions_kknn1 <- prediction(kknnFit1, testSet_all, se.fit = FALSE)
str(predictions_kknn1)
plot(predictions_kknn1$galaxysentiment,testSet_all$galaxysentiment)
cmKKNN <- confusionMatrix(predictions_kknn1$fitted,testSet_all$galaxysentiment)
cmKKNN
#%%%%%%%%%%%%%%%%%%%%%Predictions with RFE Data%%%%%%%%%%%%%%%%%%%%%
#testSet <- testSet_RFE

predictions_c52 <- prediction(c5Fit2, testSet_RFE, se.fit = FALSE)
str(predictions_c52)
plot(predictions_c52$fitted,testSet_RFE$galaxysentiment)
cmC5_rfe <- confusionMatrix(predictions_c52$fitted,testSet_RFE$galaxysentiment)
cmC5_rfe
hist(as.numeric(predictions_c52$fitted))
#--------------------------------------------------
predictions_rf2 <- prediction(rfFit2, testSet_RFE, se.fit = FALSE)
str(predictions_rf2)
plot(predictions_rf2$fitted,testSet_RFE$galaxysentiment)
#--------------------------------------------------
predictions_svm2 <- prediction(svmFit2, testSet_RFE, se.fit = FALSE)
str(predictions_svm2)
plot(predictions_svm2$fitted,testSet_RFE$galaxysentiment)
#--------------------------------------------------
predictions_kknn2 <- prediction(kknnFit2, testSet_RFE, se.fit = FALSE)
str(predictions_kknn2)
plot(predictions_kknn2$fitted,testSet_RFE$galaxysentiment)
#%%%%%%%%%%%%%%%%%%%%%Predictions with NZV Data%%%%%%%%%%%%%%%%%%%%%
#testSet <- testSet_NZV

predictions_c53 <- prediction(c5Fit3, testSet_NZV, se.fit = FALSE)
str(predictions_c53)

#--------------------------------------------------
predictions_rf3 <- prediction(rfFit3, testSet_NZV, se.fit = FALSE)
str(predictions_rf3)
plot(predictions_rf3$fitted,testSet_NZV$galaxysentiment)
#--------------------------------------------------
predictions_svm3 <- prediction(svmFit3, testSet_NZV, se.fit = FALSE)
str(predictions_svm3)
plot(predictions_svm3$fitted,testSet_NZV$galaxysentiment)
#--------------------------------------------------
predictions_kknn3 <- prediction(kknnFit3, testSet_NZV, se.fit = FALSE)
str(predictions_kknn3)
plot(predictions_kknn3$fitted,testSet_NZV$galaxysentiment)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%Predictions for the Large-Matrix with C5.0 model
#%%%%%%from  the RFE data-set

new_data <- read.csv("../mary6sentOutput/LargeMatrix.csv")
new_data$iphonesentiment <- as.factor(new_data$iphonesentiment)
str(new_data)
summary(new_data)
head(new_data)
tail(new_data)
sum(is.null(new_data))

predictions_c5p2_ND <- prediction(c5Fit2, new_data, se.fit = FALSE)
predicted <-select(predictions_c5p2_ND, -c(se.fitted,iphonesentiment))
str(predicted)
hist(as.numeric(predicted$fitted))
summary(as.numeric(predicted$fitted))

predicted$fitted
predicted$final_pred <- as.numeric(predicted$fitted)
pred0 <- predicted[ which (final_pred==0),]
count(pred0)

pred1 <- predicted[ which (final_pred==1),]
count(pred1)

pred2 <- predicted[ which (final_pred==2),]
count(pred2)

pred3 <- predicted[ which (final_pred==3),]
count(pred3)

pred4 <- predicted[ which (final_pred==4),]
count(pred4)

pred5 <- predicted[ which (final_pred==5),]
count(pred5)

#sqldf("select distinct(x) from predicted$final_pred")

pred6 <- predicted[ which (final_pred==6),]
count(pred6)

hist(as.numeric(predicted$fitted))

stopCluster(cl)

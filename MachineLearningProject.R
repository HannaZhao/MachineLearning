library(caret)

fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
fileDes <- "C:/Users/zhao_h/Documents/R_WD/pml-training.csv"

download.file(fileUrl, fileDes)

training <- read.csv(fileDes, na.strings = c("","NA"))

fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
fileDes <- "C:/Users/zhao_h/Documents/R_WD//pml-testing.csv"

download.file(fileUrl, fileDes)
testing <- read.csv(fileDes, na.strings = c("","NA"))

col_mark <- colSums(is.na(training))/dim(training)[1]<0.90

Pre_training <- training[,col_mark]
Pre_training <- Pre_training[,-1]

testing <- testing[, col_mark]
testing <- testing[, -1]

set.seed(333)
indTrain <- createDataPartition(y = Pre_training$classe, p=0.6, list = FALSE)
Part_training <-Pre_training[indTrain,]
Part_testing <-Pre_training[-indTrain,]


t1 <- system.time(mod1<- train(classe ~ ., method = "rpart", data = Part_training))
confusionMatrix(Part_testing$classe, predict(mod1,Part_testing))

t2<- system.time(mod2<- train(classe ~ ., method = "rf", data = Part_training, prox=TRUE))
confusionMatrix(Part_testing$classe, predict(mod2,Part_testing))

t3<- system.time(mod3<- train(classe ~ ., method = "gbm", data = Part_training))
confusionMatrix(Part_testing$classe, predict(mod3,Part_testing))

t4<- system.time(mod4<- train(classe ~ ., method = "rf", data = Part_training, prox=TRUE, ntree=20))
confusionMatrix(Part_testing$classe, predict(mod4,Part_testing))


predict(mod1, testing)
predict(mod2, testing)
predict(mod3, testing)
predict(mod4, testing)




library(neuralnet)
library(caret)

#######################
#NEURAL NETWORK MODEL
#######################


#Set up model dataframe from the online available dataset
WineML <- read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', sep=",", dec=".",header=FALSE)
head(WineML)

#Label variables and make all variables as.numeric
colnames(WineML) <- c("winery","alcohol","malic","ash","ash_alcalinity","Mg","phenols","flavanoids","nonflavanoids","proanthocyanins","color_intensity","hue","od280","proline") 
str(WineML)
WineML$Mg <- as.numeric(WineML$Mg)
WineML$proline <- as.numeric(WineML$proline)
WineML$winery <- as.numeric(WineML$winery) #change this to factor after dataset normalization


#Dataset Max-Min Normalization for neuralnet model
normalize <- function(x) { 
    return ((x - min(x)) / (max(x) - min(x)))
}

maxmindf <- as.data.frame(lapply(WineML[2:14], normalize))

maxmindf <- cbind(winery=WineML$winery, maxmindf)
maxmindf$winery <- as.factor(maxmindf$winery) #change this to factor (outcome variable)
str(maxmindf)

#Partition training (70%) and testing (30%) datasets with normalized representation of the 'winery' outcome variable
indexes <- createDataPartition(maxmindf$winery,
                               times = 1,
                               p = 0.7,
                               list = FALSE)
trainset <- maxmindf[indexes,]
testset <- maxmindf[-indexes,]

#Inspect the Outcome variable ($winery) proportions
prop.table(table(maxmindf$winery))
prop.table(table(trainset$winery))
prop.table(table(testset$winery))


#Establish neuralnet model on <trainset> data: specify Outcome variable (winery), configure hidden neurons, other parameters
nn <- neuralnet(winery ~ ., data=trainset, hidden=c(2,1), linear.output=FALSE, threshold=0.01)
nn$result.matrix
plot(nn)


#Test neuralnet model on <testset> data (excluding the outcome variable obviously)
temp_test <- subset(testset, select=c("alcohol","malic","ash","ash_alcalinity","Mg","phenols","flavanoids","nonflavanoids",
                                      "proanthocyanins","color_intensity","hue","od280","proline"))

#Evaluate neuralnet model classification performance
nn.results <- compute(nn, temp_test)
results <- data.frame(nn.results$net.result)
colnames(results) <- c("1","2","3") #corresponding to votes/score for winery#1,#2 and #3, respectively
results$max <- colnames(results)[max.col(results,ties.method="first")] #attribute highest vote to corresponding outcome
results$prediction = ifelse(results$max==1, 1, ifelse(results$max==2,2, ifelse(results$max==3,3,NA))) #workaround for the above column 'char' structure
results$prediction <- as.factor(results$prediction)
str(results)

confusionMatrix(results$prediction, testset$winery)




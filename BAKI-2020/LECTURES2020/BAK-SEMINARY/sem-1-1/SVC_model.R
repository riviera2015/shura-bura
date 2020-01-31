#-------------------------------------------------------------------------------
# Setup

require(dplyr)
require(e1071)

load(file="images_formatted.Rdata")
D <- out; rm(out)

# Scale the data. Note that scale returns a matrix
D <- scale(D)

# Reproducibility
set.seed(500)

# Index to be sampled for train/test split
index <- 1:nrow(D)

# Number of cross validation runs
K <- 20

#-------------------------------------------------------------------------------
# SVC model on real data

# Assign variable names
colnames(D) <- paste("pixel_", 1:4096, sep = "")

# Convert images to dataframe and add label information as a factor
D <- D %>% tbl_df() %>% mutate(label = as.factor(y_df))

accuracy_first <- list()
for(i in 1:K)
{
    # Split data into a train and test set.
    testindex <- sample(index, trunc(length(index)/3))
    # Test set. 1/3 of full data.
    testset <- D[testindex, ]
    # Train set. 2/3 of full data.
    trainset <- D[-testindex, ]
    # Solution to test set
    actual_sol <- testset %>% select(label) %>% pull()
    
    # SVC model on real data: train
    svc.model <- svm(label ~., data = trainset, cost = 1, kernel = "linear")
    # SVC model on real data: classify test set
    svc.pred  <- predict(svc.model, testset[, -4097])
    
    # Accuracy of SVC on real data
    accuracy_first[[i]] <- sum(svc.pred == actual_sol)/length(actual_sol)
    # Print accuracy of current iteration
    print(accuracy_first[[i]])
}

# Average accuracy of the model
print(mean(as.numeric(accuracy_first)))

rm(testset, trainset, actual_sol, svc.model, svc.pred, i)

#-------------------------------------------------------------------------------
# SVC model on PCA (30 principal components)

# Run PCA on data
pca_faces <- D %>% select(-label) %>% as.matrix() %>% prcomp()
plot((cumsum(pca_faces$sdev^2)/sum(pca_faces$sdev^2))[1:30], type="o", xlab = "Eigenvalue #", ylab = "% of total variance", main = "Percentage of total variance explained", col = "blue")
plot((pca_faces$sdev^2)[1:30], type="o", xlab = "Eigenvalue #", ylab = "Magnitude", main = "Magnitude of eigenvalues", col = "green")

# % of total variance explained with 30 components
(cumsum(pca_faces$sdev^2)/sum(pca_faces$sdev^2))[30]

accuracy_second <- list()
for(i in 1:K)
{
    # Split data into a train and test set
    testindex <- sample(index, trunc(length(index)/3))
    # Test set from PCA
    testset <- as.matrix(D[testindex, -4097]) %*% pca_faces$rotation[, 1:30]
    colnames(testset) <- paste("PC_", 1:30, sep="")
    # Solutions to test set
    actual_sol <- D[testindex, 4097] %>% pull(label)
    # Train set from PCA
    trainset <- as.matrix(D[-testindex, -4097]) %*% pca_faces$rotation[, 1:30]
    trainset <- as.data.frame(trainset)
    colnames(trainset) <- paste("PC_", 1:30, sep="")
    # Add label information to test set
    trainset$label <- D[-testindex, 4097] %>% pull(label)
    
    # SVC model on PCA data
    svc.model <- svm(label ~., data = trainset, cost = 1, kernel = "linear")
    svc.pred  <- predict(svc.model, testset)

    # Accuracy of SVM on real data
    accuracy_second[[i]] <- sum(svc.pred == actual_sol)/length(actual_sol)
    print(accuracy_second[[i]])
}

print(mean(as.numeric(accuracy_second)))

#-------------------------------------------------------------------------------
# Comparison between the two accuracies

results <- data.frame(data_model = as.numeric(accuracy_first),
                      pca_model = as.numeric(accuracy_second))

results %>% summarise_all(mean)
results %>% summarise_all(sd)

require(reshape2)
require(ggplot2)
ggplot(melt(results), aes(x = variable, y = value, fill = variable)) +
    geom_boxplot() +
    ggtitle("Comparison between accuracies of the two models") +
    ylab("Accuracy") +
    xlab("Model")

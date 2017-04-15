#Include packages
library(ggplot2)
library(caret)
library(ROCR)
library(pROC)

#Setting file name
filename <- "VoterPref.csv"

#Reading csv into dataframe and create success class
df <- read.csv(filename, header = TRUE)
SUCCESS <- ifelse((df$PREFERENCE == 'Against'), 1, 0)
df <- cbind(df, SUCCESS)
attach(df)
head(df)

#Setting seed value and partition by random sampling
set.seed(123457)
train <- sample(nrow(df), 0.7*nrow(df))
df_train <- df[train,]
df_val <- df[-train,]
str(df_train)
str(df_val)

#Logistic regression on training data set
fit <- glm(SUCCESS ~ AGE + INCOME + GENDER, data = df_train, family = "binomial")
summary(fit)

#Classification and ploting Confusion matrix for training and validation data sets
df_train$PREDICTED_PROBABILITY <- predict.glm(fit, df_train, type = "response")
df_train$PREDICTED_OUTCOME <- ifelse((df_train$PREDICTED_PROBABILITY > 0.5), 1, 0)

df_val$PREDICTED_PROBABILITY <- predict.glm(fit, df_val, type = "response")
df_val$PREDICTED_OUTCOME <- ifelse((df_val$PREDICTED_PROBABILITY > 0.5), 1, 0)

head(df_train)
head(df_val)

conf_matrix_train <- confusionMatrix(df_train$PREDICTED_OUTCOME, df_train$SUCCESS)
conf_matrix_val <- confusionMatrix(df_val$PREDICTED_OUTCOME, df_val$SUCCESS)
conf_matrix_train
conf_matrix_val

#Plot ROC Curves with pROC package for training and validation datasets
roc_curve_train <- roc(df_train$SUCCESS, df_train$PREDICTED_PROBABILITY)
roc_curve_val <- roc(df_val$SUCCESS, df_val$PREDICTED_PROBABILITY)

plot.roc(roc_curve_train, legacy.axes = TRUE, col = "BLUE")
plot.roc(roc_curve_val, legacy.axes = TRUE, add = TRUE, col = "GREEN")

#Plot cutoff vs accuracy for training dataset
cutoff <- seq(0, 1, length = 10000)
accuracy <- numeric(10000)
acc_plot_table_train <- data.frame(CUTOFF = cutoff , ACCURACY = accuracy)

for(index in 1:10000) {
  pred <- factor(ifelse((df_train$PREDICTED_PROBABILITY > cutoff[index]), 1, 0))
  true_positives <- sum(pred == 1 & df_train$SUCCESS == 1)
  true_negatives <- sum(pred == 0 & df_train$SUCCESS == 0)
  total <- length(pred)
  acc_plot_table_train$ACCURACY[index] <- (true_positives + true_negatives)/total
}
ggplot(data = acc_plot_table_train, mapping = aes(x = CUTOFF, y = ACCURACY, col)) + geom_line(size = 1)

#Plot cutoff vs accuracy for validation dataset
acc_plot_table_val <- data.frame(CUTOFF = cutoff , ACCURACY = accuracy)

for(index in 1:10000) {
  pred <- factor(ifelse((df_val$PREDICTED_PROBABILITY > cutoff[index]), 1, 0))
  true_positives <- sum(pred == 1 & df_val$SUCCESS == 1)
  true_negatives <- sum(pred == 0 & df_val$SUCCESS == 0)
  total <- length(pred)
  acc_plot_table_val$ACCURACY[index] <- (true_positives + true_negatives)/total
}
ggplot(data = acc_plot_table_val, mapping = aes(x = CUTOFF, y = ACCURACY, col)) + geom_line(size = 1)

#Display cutoff with maximum accuracy for training dataset
acc_plot_table_train[which(acc_plot_table_train$ACCURACY == max(acc_plot_table_train$ACCURACY)),]
#Rounding off to 3 significant digits
max_acc_cutoff_train <- signif(mean(acc_plot_table_train[which(acc_plot_table_train$ACCURACY 
                                                               == max(acc_plot_table_train$ACCURACY)), 1]), 3)

max_acc_cutoff_train

#Display cutoff with maximum accuracy for validation dataset
acc_plot_table_val[which(acc_plot_table_val$ACCURACY == max(acc_plot_table_val$ACCURACY)),]
#Rounding off to 3 significant digits
max_acc_cutoff_val <- signif(mean(acc_plot_table_val[which(acc_plot_table_val$ACCURACY 
                                                           == max(acc_plot_table_val$ACCURACY)), 1]), 3)
max_acc_cutoff_val

#Display accuracy in validation dataset for max accuracy cutoff in training dataset
#Rounding off to 3 significant digits
acc_val <- signif(mean(acc_plot_table_val[which(signif(acc_plot_table_val$CUTOFF, 3) == 
                                                  max_acc_cutoff_train),2]), 3)
acc_val

#Misclassification cost estimation
cost_false_positive <- 4
cost_false_negative <- 1
cutoff1 <- seq(0, 1, length = 100)
cost <- numeric(100)
mcdf <- data.frame(CUTOFF = cutoff1, COST = cost)
for(index in 1:100) {
  pred1 <- factor(ifelse((df_train$PREDICTED_PROBABILITY > cutoff1[index]), 1, 0))
  false_positives <- sum(pred1 == 1 & df_train$SUCCESS == 0)
  false_negatives <- sum(pred1 == 0 & df_train$SUCCESS == 1)
  mc_cost <- (false_positives * cost_false_positive) + (false_negatives * cost_false_negative)
  mcdf$COST[index] <- mc_cost
}
#Find cutoff with minimum misclassification cost in training dataset
cutoff_mincost_train <- signif(mcdf[which(mcdf$COST == min(mcdf$COST)),], 3)
cutoff_mincost_train

#Calculate misclassification cost for validation dataset
pred2 <- factor(ifelse((df_val$PREDICTED_PROBABILITY > cutoff_mincost_train[,1]), 1, 0))
false_positives <- sum(pred2 == 1 & df_val$SUCCESS == 0)
false_negatives <- sum(pred2 == 0 & df_val$SUCCESS == 1)
mc_cost_val <- (false_positives * cost_false_positive) + (false_negatives * cost_false_negative)
mc_cost_val

#Compare cutoffs
cutoff_mincost_vs_maxacc_cutoff <- signif(mcdf[which(signif(mcdf$CUTOFF, 2) == 
                                                       signif(max_acc_cutoff_train, 2)),], 3)
compare_df <- rbind(cutoff_mincost_vs_maxacc_cutoff, cutoff_mincost_train)
rownames(compare_df) <- c("Max Acc", "Min Cost")
compare_df

#Gains Chart training dataset
gs_train <- data.frame(PREDICTED = df_train$PREDICTED_OUTCOME, ACTUAL = df_train$SUCCESS)
gs_train_sorted <- gs_train[order(-gs_train$PREDICTED),]
gs_train_sorted$GAINS <- cumsum(gs_train_sorted$ACTUAL)
plot(gs_train_sorted$GAINS,type = "n", main = "Training Data Gains Chart",
     xlab = "Number of Cases", ylab = "Cumulative Success")
lines(gs_train_sorted$GAINS)
abline(0,sum(gs_train_sorted$ACTUAL)/nrow(gs_train_sorted),lty = 2, col = "red")

#Gains Chart test datasey
gs_val <- data.frame(PREDICTED = df_val$PREDICTED_OUTCOME, ACTUAL = df_val$SUCCESS)
gs_val_sorted <- gs_val[order(-gs_val$PREDICTED),]
gs_val_sorted$GAINS <- cumsum(gs_val_sorted$ACTUAL)
plot(gs_val_sorted$GAINS,type = "n", main = "Validation Data Gains Chart",
     xlab = "Number of Cases", ylab = "Cumulative Success")
lines(gs_val_sorted$GAINS)
abline(0,sum(gs_val_sorted$ACTUAL)/nrow(gs_val_sorted),lty = 2, col = "red")







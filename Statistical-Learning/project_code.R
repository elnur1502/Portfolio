library(splitTools)
library(pROC)
library(ROSE)
library(ggplot2)
library(cowplot)
library(dplyr)
library(Rmisc)
library(corrplot)
library(MASS)
library(caret)
library(regclass)
library(fastDummies)
library(class)
library(glmnet)
library(outliers)
library(mlr3measures)
library(pracma)

setwd("...")

# https://archive.ics.uci.edu/ml/machine-learning-databases/00222/
# http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#
# data <- read.csv('data/bank-full.csv', sep=';', na.strings="unknown")
data <- read.csv('data/bank-additional-full.csv', sep=';', 
                 na.strings=c("unknown"))


# length and info
length(data)
summary(data)

# class disbalance
table(data$y)

#group features
bin_feats <- c("default", "housing", "loan", "contact")
ordinal_feats <- c("education")
nominal_feats <- c("job", "marital",
                   "month", "day_of_week", "poutcome")
continuous_feats <- c("age", "duration", "campaign", "pdays", "previous",
                      "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                      "euribor3m", "nr.employed")

# unique values of cat feats
for (feat in c(nominal_feats, ordinal_feats)) {
  print(feat)
  print(unique(data[, feat]))
  print("________________")
}

# a part for data viz
# prepare some data for analysis
# convert cat feats to factor 
data$y <- as.factor(data$y)
data$marital <- factor(data$marital, levels = c("single", "divorced", 
                                                "married"), ordered = FALSE)
data$education <- factor(data$education, levels = c('illiterate',
                                                    'basic.4y','basic.6y',
                                                    'basic.9y','high.school',
                                                    'professional.course',
                                                    'university.degree'), 
                         ordered = TRUE)
data$default <- as.factor(data$default)
data$housing <- as.factor(data$housing)
data$loan <- as.factor(data$loan)
data$contact <- as.factor(data$contact)
#data$month <- factor(data$month, levels = c("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), ordered = TRUE)
#data$month <- as.integer(factor(data$month, levels = c("jan", "feb", "mar", "apr", "may", 
#                                                       "jun", "jul", "aug", "sep", "oct", 
#                                                       "nov", "dec"), ordered = TRUE))
data$day_of_week <- factor(data$day_of_week, 
                           levels = c("mon", "tue", "wed", "thu", "fri"), 
                           ordered = FALSE)
data$poutcome <- factor(data$poutcome, levels = c("nonexistent", "failure", 
                                                  "success"), ordered = FALSE)
data$job <- factor(data$job, levels = c('admin.','blue-collar','entrepreneur',
                                        'housemaid', 'management','retired',
                                        'self-employed','services',
                                        'student','technician','unemployed'), 
                   ordered = FALSE)
data$month <- factor(data$month, levels = c("jan", "feb", "mar", "apr", "may", 
                                            "jun", "jul", "aug", "sep", "oct", 
                                            "nov", "dec"), ordered = TRUE)
# adding year
data$year <- NA
year = 2008
for (i in 1:(nrow(data)-1)) {
  data[i, 'year'] <- year
  if (data[i, 'month'] > data[i+1, 'month']) {
    year <- year + 1
  } 
}
data[i+1, 'year'] <- year

# adding date
data$date <- paste(as.character(data$year), "01", 
                   as.character(as.numeric(data$month)), 
                   sep = "-")
data$date <- as.Date(data$date)

data$month

# adding season
get_season <- function(x) {
  if (is.element(x, c("dec", "jan", "feb"))) {
    season <- 'win'
  } else if (is.element(x, c("mar", "apr", "may"))) {
    season <- 'spr'
  } else if (is.element(x, c("jun", "jul", "aug"))) {
    season <- 'sum'
  } else {
    season <- 'aut'
  }
  return(season)
}

# data$season <- sapply(data$month, get_season)
# data$season <- factor(data$season, levels = c('win','spr','sum',
#                                         'aut'), ordered = FALSE)

# Working with missing values
ceiling(colSums(is.na(data)) / nrow(data) * 100)

calc_mode <- function(x){
  
  # List the distinct / unique values
  distinct_values <- unique(x)
  
  # Count the occurrence of each distinct value
  distinct_tabulate <- tabulate(match(x, distinct_values))
  
  # Return the value with the highest occurrence
  distinct_values[which.max(distinct_tabulate)]
}

fill_na_tr_tst <- function(tr, tst, feats) {
  tr_filled <- tr
  tst_filled <- tst
  for (feat in feats) {
    mode_val <- calc_mode(tr[, feat])
    tr_filled[is.na(tr_filled[, feat]), feat] <- mode_val
    tst_filled[is.na(tst_filled[, feat]), feat] <- mode_val
  }
  return(list(train = tr_filled, test = tst_filled))
}

data$month <- factor(data$month, levels = c("jan", "feb", "mar", "apr", "may", 
                                            "jun", "jul", "aug", "sep", "oct", 
                                            "nov", "dec"), ordered = FALSE)

# Treat month as integer for future prediction
#data$month <- as.integer(factor(data$month, levels = c("jan", "feb", "mar", "apr", "may", 
#                                                       "jun", "jul", "aug", "sep", "oct", 
 #                                                      "nov", "dec"), ordered = TRUE))
#data$month <- factor(data$month, levels = c("jan", "feb", "mar", "apr", "may", 
#                                                       "jun", "jul", "aug", "sep", "oct", 
#                                                       "nov", "dec"), ordered = FALSE)

# poly feats
#data$year2 = data$year^2
#data$year3 = data$year^3

#data$month2 = data$month^2
#data$month3 = data$month^2

# remove leak feats
data <- subset(data, select = -c(duration))

# remove useless feats
default <- data$default
default[is.na(default)] <- "no"
data <- subset(data, select = -c(default, month, date))

# update groups of feats
continuous_feats <- continuous_feats[! continuous_feats %in% c('duration')]
continuous_feats <- c(continuous_feats, "year")
nominal_feats <- nominal_feats[! nominal_feats %in% c('month')]
# nominal_feats <- c(nominal_feats, "season")
bin_feats <- bin_feats[! bin_feats %in% c('default')]


summary(data)
sum(is.na.data.frame(data))

# time series train test split
nrow(data)
cut_idx = round(nrow(data) * 0.8)
train <- data[0:cut_idx, ]
test <- data[(cut_idx+1):nrow(data), ]

table(train$y) / nrow(train) * 100
table(test$y) / nrow(test) * 100

# fill NA with mode
missing_feats <- c("job", "education", "marital", "housing", "loan")
filled_dfs <-fill_na_tr_tst(train, test, missing_feats)
train <- filled_dfs$train
test <- filled_dfs$test

# remove outliers from train set
train <- train[default[0:cut_idx] != "yes", ]
max_campaign <- quantile(train$campaign)[4] + 1.5*IQR(train$campaign)
train <- train[train$campaign <= max_campaign, ]

# create a validation set
#val_cut_idx = nrow(train) - nrow(test) - 1
val_cut_idx = round(nrow(train) * 0.85)
val_train <- train[0:val_cut_idx, ]
val_val <- train[(val_cut_idx+1):nrow(train), ]

# stratified train test split
#set.seed(228)
#inds <- partition(data$y, p = c(train = 0.8, test = 0.2))
#train <- data[inds$train, ]
#test <- data[inds$test, ]

adjust_threshold <- function(truth, proba) {
  best_val <- 0
  best_score <- 0
  for (val in linspace(0, 1, 101)) {
    preds <- ifelse(proba > val, "yes", "no")
    preds <- factor(preds, levels=c("no", "yes"))
    score <- fbeta(truth, preds, positive="yes", 
                   beta = 2)
    if (is.na(score)) {
      score <- 0
    }
    if (score > best_score) {
      best_score <- score
      best_val <- val
    }
  }
  return(best_val)
}

# Logistic regression
mod.out <- glm(y ~., data=val_train, family = binomial)
probabilities <- predict(mod.out, val_val, type="response")
threshold <- adjust_threshold(val_val$y, probabilities)
threshold

mod.out <- glm(y ~., data=train, family = binomial)
summary(mod.out)
probabilities <- predict(mod.out, test, type="response")
preds <- ifelse(probabilities > threshold, "yes", "no")
preds <- factor(preds, levels=c("no", "yes"))

roc_score <- roc(test$y, probabilities) #AUC score
f2_score <- fbeta(test$y, preds, positive="yes", beta = 2)
f2_score
plot(roc_score ,main ="ROC curve -- Logistic Regression ")

roc_score

confusionMatrix(data=preds, reference = test$y)

# oversampling
train_over <- ovun.sample(y~., data = train, method = "over")$data
val_train_over <- ovun.sample(y~., data = val_train, method = "over")$data

mod.out <- glm(y ~., data=val_train_over, family = binomial)
probabilities <- predict(mod.out, val_val, type="response")
threshold <- adjust_threshold(val_val$y, probabilities)
threshold

mod.out <- glm(y ~., data=train_over, family = binomial)
summary(mod.out)

probabilities <- predict(mod.out, test, type="response")
preds <- ifelse(probabilities > threshold, "yes", "no")
preds <- as.factor(preds)

roc_score=roc(test$y, probabilities) #AUC score
f2_score <- fbeta(test$y, preds, positive="yes", beta = 2)
f2_score
plot(roc_score ,main ="ROC curve -- Logistic Regression ")

roc_score

confusionMatrix(data=preds, reference = test$y)

# VIF
mod.out <- glm(y ~.-pdays-previous, data=val_train, family = binomial)
probabilities <- predict(mod.out, val_val, type="response")

threshold <- adjust_threshold(val_val$y, probabilities)
threshold

mod.out <- glm(y ~.-pdays-previous, data=train, family = binomial)
# bad score if exclude all
#mod.out <- glm(y ~.-euribor3m-emp.var.rate-pdays-nr.employed-previous-year, data=train, family = binomial)
VIF(mod.out)
print(5^0.5)
print(10^0.5)

summary(mod.out)
probabilities <- predict(mod.out, test, type="response")
preds <- ifelse(probabilities > threshold, "yes", "no")
preds <- as.factor(preds)

roc_score=roc(test$y, probabilities) #AUC score
plot(roc_score ,main ="ROC curve -- Logistic Regression ")

f2_score <- fbeta(test$y, preds, positive="yes", beta = 2)
f2_score

roc_score

confusionMatrix(data=preds, reference = test$y)

# Backward Stepwise Selection
mod.out <- glm(y ~., data=train, family = binomial)
b_step <- step(mod.out, direction= "backward", scope=formula(mod.out), trace=0)
b_step$anova
b_step$coefficients

mod.out <- glm(y ~.-previous-housing-campaign-loan-age, data=train_val, family = binomial)
probabilities <- predict(mod.out, val_val, type="response")

threshold <- adjust_threshold(val_val$y, probabilities)
threshold

mod.out <- glm(y ~.-previous-housing-campaign-loan-age, data=train, family = binomial)
summary(mod.out)
probabilities <- predict(mod.out, test, type="response")
preds <- ifelse(probabilities > threshold, "yes", "no")
preds <- as.factor(preds)

roc_score=roc(test$y, probabilities) #AUC score
plot(roc_score ,main ="ROC curve -- Logistic Regression ")

roc_score

f2_score <- fbeta(test$y, preds, positive="yes", beta = 2)
f2_score

confusionMatrix(data=preds, reference = test$y)

# Forward Stepwise Selection
mod.out <- glm(y ~., data=train, family = binomial)
mod.intercept <- glm(y ~ 1, data=train, family = binomial)
f_step <- step(mod.intercept, direction= "forward", scope=formula(mod.out), 
               trace=0)
f_step$anova
f_step$coefficients
# The same results

# LDA
# Checking normality assumption
par(mfrow=c(2, 4))
for (feat in continuous_feats[1:4]) {
  for (cl_lab in c("yes", "no")) {
    init_arr <- data[data$y == cl_lab, feat]
    size <- min(5000, length(init_arr))
    arr <- sample(init_arr, size)
    res <- shapiro.test(arr)
    p_val <- res[2]$p.value
    p_val <- round(p_val, 2)
    qqnorm(arr, pch = 1, frame = FALSE,
           main=sprintf("%s. y = %s. Sh.-W. norm test. p-value = %s", 
                        feat, cl_lab, p_val))
    qqline(arr, col = "steelblue", lwd = 2)
    #mtext(sprintf("Shapiro-Wilk normality test. p-value = %s", p_val), side = 3, 
    #line = - 4, outer = TRUE)
  }
}
par(mfrow=c(2, 4))
for (feat in continuous_feats[5:8]) {
  for (cl_lab in c("yes", "no")) {
    init_arr <- data[data$y == cl_lab, feat]
    size <- min(5000, length(init_arr))
    arr <- sample(init_arr, size)
    res <- shapiro.test(arr)
    p_val <- res[2]$p.value
    p_val <- round(p_val, 2)
    qqnorm(arr, pch = 1, frame = FALSE,
           main=sprintf("%s. y = %s. Sh.-W. norm test. p-value = %s", 
                        feat, cl_lab, p_val))
    qqline(arr, col = "steelblue", lwd = 2)
    #mtext(sprintf("Shapiro-Wilk normality test. p-value = %s", p_val), side = 3, 
    #line = - 4, outer = TRUE)
  }
}

par(mfrow=c(2, 4))
for (feat in continuous_feats[9:10]) {
  for (cl_lab in c("yes", "no")) {
    init_arr <- data[data$y == cl_lab, feat]
    size <- min(5000, length(init_arr))
    arr <- sample(init_arr, size)
    res <- shapiro.test(arr)
    p_val <- res[2]$p.value
    p_val <- round(p_val, 2)
    qqnorm(arr, pch = 1, frame = FALSE,
           main=sprintf("%s. y = %s. Sh.-W. norm test. p-value = %s", 
                        feat, cl_lab, p_val))
    qqline(arr, col = "steelblue", lwd = 2)
    #mtext(sprintf("Shapiro-Wilk normality test. p-value = %s", p_val), side = 3, 
    #line = - 4, outer = TRUE)
  }
}

x_prepared <- NULL

perform_bc_transform <- function(x) {
  if (min(x) <= 0) {
    x_prepared <<- x - min(x) + 1} else {
      x_prepared <<- x
    }
  search_results <- boxcox(lm(x_prepared ~ 1), plotit = FALSE)
  lambda <- search_results$x[which.max(search_results$y)]
  if (lambda != 0) {x_transformed <- (x_prepared^lambda - 1) / lambda}
  else {x_transformed <- log(x_prepared)}
  return(x_transformed)
}

data_bc_transform <- data
data_bc_transform$year <- data_bc_transform$year - 2008
data_bc_transform[, continuous_feats] <-
  sapply(data_bc_transform[, continuous_feats], perform_bc_transform)

train_bc <- data_bc_transform[0:cut_idx, ]
test_bc <- data_bc_transform[(cut_idx+1):nrow(data), ]

filled_dfs_bc <- fill_na_tr_tst(train_bc, test_bc, missing_feats)
train_bc <- filled_dfs_bc$train
test_bc <- filled_dfs_bc$test

# 
# par(mfrow=c(2, 4))
# for (feat in continuous_feats[1:4]) {
#   for (cl_lab in c("yes", "no")) {
#     init_arr <- data_bc_transform[data_bc_transform$y == cl_lab, feat]
#     size <- min(5000, length(init_arr))
#     arr <- sample(init_arr, size)
#     res <- shapiro.test(arr)
#     p_val <- res[2]$p.value
#     p_val <- round(p_val, 2)
#     qqnorm(arr, pch = 1, frame = FALSE,
#            main=sprintf("%s. y = %s. Sh.-W. norm test. p-value = %s", 
#                         feat, cl_lab, p_val))
#     qqline(arr, col = "steelblue", lwd = 2)
#   }
# }
# 
# par(mfrow=c(2, 4))
# for (feat in continuous_feats[5:8]) {
#   for (cl_lab in c("yes", "no")) {
#     init_arr <- data_bc_transform[data_bc_transform$y == cl_lab, feat]
#     size <- min(5000, length(init_arr))
#     arr <- sample(init_arr, size)
#     res <- shapiro.test(arr)
#     p_val <- res[2]$p.value
#     p_val <- round(p_val, 2)
#     qqnorm(arr, pch = 1, frame = FALSE,
#            main=sprintf("%s. y = %s. Sh.-W. norm test. p-value = %s", 
#                         feat, cl_lab, p_val))
#     qqline(arr, col = "steelblue", lwd = 2)
#   }
# }
# 
# par(mfrow=c(2, 4))
# for (feat in continuous_feats[9:10]) {
#   for (cl_lab in c("yes", "no")) {
#     init_arr <- data_bc_transform[data_bc_transform$y == cl_lab, feat]
#     size <- min(5000, length(init_arr))
#     arr <- sample(init_arr, size)
#     res <- shapiro.test(arr)
#     p_val <- res[2]$p.value
#     p_val <- round(p_val, 2)
#     qqnorm(arr, pch = 1, frame = FALSE,
#            main=sprintf("%s. y = %s. Sh.-W. norm test. p-value = %s", 
#                         feat, cl_lab, p_val))
#     qqline(arr, col = "steelblue", lwd = 2)
#   }
# }
# 
# par(mfrow=c(1, 1))
# 
# train_bc <- data_bc_transform[0:cut_idx, ]
# test_bc <- data_bc_transform[cut_idx:nrow(data), ]

mod.out <- lda(y ~.-year, family = "binomial", data = val_train)
probabilities <- predict(mod.out, val_val, type="response")
probabilities <- probabilities$posterior[,2]
threshold <- adjust_threshold(val_val$y, probabilities)
threshold

mod.out <- lda(y ~., family = "binomial", data = train)
summary(mod.out)
probabilities <- predict(mod.out, test, type="response")
probabilities <- probabilities$posterior[,2]
preds <- ifelse(probabilities > threshold, "yes", "no")
preds <- as.factor(preds)

roc_score=roc(test$y, probabilities) #AUC score
plot(roc_score ,main ="ROC curve -- Logistic Regression ")

roc_score

f2_score <- fbeta(test$y, preds, positive="yes", beta = 2)
f2_score

confusionMatrix(data=preds, reference = test$y)

# QDA
val_train_bc <- train_bc[0:val_cut_idx, ]
val_val_bc <- train_bc[(val_cut_idx+1):nrow(train_bc), ]
mod.out <- qda(y ~., family = "binomial", data = val_train)
probabilities <- predict(mod.out, val_val_bc, type="response")
probabilities <- probabilities$posterior[,2]
threshold <- adjust_threshold(val_val_bc$y, probabilities)
threshold


mod.out <- qda(y ~.,family = "binomial", data = train)

summary(mod.out)
probabilities <- predict(mod.out, test, type="response")
probabilities <- probabilities$posterior[,2]
preds <- ifelse(probabilities > threshold, "yes", "no")
preds <- as.factor(preds)

roc_score=roc(test$y, probabilities) #AUC score
plot(roc_score ,main ="ROC curve -- Logistic Regression ")

roc_score

confusionMatrix(data=preds, reference = test$y)

# KNN
data_for_dummies <- rbind(train, test)
data_dummies <- dummy_cols(data_for_dummies, select_columns=c(bin_feats, ordinal_feats, nominal_feats),
                           remove_selected_columns=TRUE)
colnames(data_dummies)[colnames(data_dummies) == "job_blue-collar"] <- "job_blue_collar"
colnames(data_dummies)[colnames(data_dummies) == "job_self-employed"] <- "job_self_employed"

train_dummies <- data_dummies[0:nrow(train), ]
test_dummies <- data_dummies[(nrow(train_dummies)+1):nrow(data_dummies), ]

val_train_dummies <- train_dummies[0:val_cut_idx, ]
val_val_dummies <- train_dummies[(val_cut_idx+1):nrow(train_dummies), ]

# oversampling
train_dummies_over <- ovun.sample(y~., data = train_dummies, method = "over")$data
train_dummies_over <- train_dummies

val_train_dummies_over <- ovun.sample(y~., data = val_train_dummies, method = "over")$data
val_train_dummies_over <- val_train_dummies

# scaling
mins <- sapply(subset(train_dummies_over, select = -c(y)), min)
maxs <- sapply(subset(train_dummies_over, select = -c(y)), max)
maxs_m_mins <- maxs - mins

train_dummies_over_sc <- sweep(subset(train_dummies_over, select = -c(y)),
      2,
      mins)

train_dummies_over_sc <- sweep(train_dummies_over_sc,
                               2,
                               maxs_m_mins, FUN='/')
train_dummies_over_sc$y <- train_dummies_over$y

test_dummies_sc <- sweep(subset(test_dummies, select = -c(y)),
                               2,
                               mins)

test_dummies_sc <- sweep(test_dummies_sc,
                               2,
                               maxs_m_mins, FUN='/')
test_dummies_sc$y <- test_dummies$y

val_train_dummies_over_sc <- sweep(subset(val_train_dummies_over, select = -c(y)),
                               2,
                               mins)

val_train_dummies_over_sc <- sweep(val_train_dummies_over_sc,
                               2,
                               maxs_m_mins, FUN='/')
val_train_dummies_over_sc$y <- val_train_dummies_over$y

val_val_dummies_sc <- sweep(subset(val_val_dummies, select = -c(y)),
                         2,
                         mins)

val_val_dummies_sc <- sweep(val_val_dummies_sc,
                         2,
                         maxs_m_mins, FUN='/')
val_val_dummies_sc$y <- val_val_dummies$y

# select k

scores <- c()
thresholds <- c()
ks <- seq(from = 5, to = 200, by = 20)
for (k in ks) {
  preds <- knn(train=subset(val_train_dummies_over_sc, select = -c(y)), 
               test=subset(val_val_dummies_sc, select = -c(y)), 
               cl = val_train_dummies_over_sc$y, k = k,
               prob = TRUE)
  probabilities <- attributes(preds)$prob
  #score <- pROC::auc(val_val_dummies_sc$y, probabilities)
  #scores <- append(scores,score)
  threshold <- adjust_threshold(val_val_dummies_sc$y, probabilities)
  preds <- ifelse(probabilities > threshold, "yes", "no")
  preds <- factor(preds, levels=c("no", "yes"))
  score  <- fbeta(val_val_dummies_sc$y, preds, positive="yes", beta = 2)
  scores <- c(scores, score)
  thresholds <- c(thresholds, threshold)
}

ks[scores == max(scores)]
threshold <- thresholds[scores == max(scores)]
max(scores)
threshold

plot(ks, scores, type='l')

preds <- knn(train=subset(train_dummies_over_sc, select = -c(y)), 
                     test=subset(test_dummies_sc, select = -c(y)), 
                     cl = train_dummies_over_sc$y, k = 85,
                     prob = TRUE)

probabilities <- attributes(preds)$prob
preds <- ifelse(probabilities > threshold, "yes", "no")
preds <- factor(preds, levels=c("no", "yes"))

roc_score=roc(test$y, probabilities) #AUC score
plot(roc_score ,main ="ROC curve -- Logistic Regression ")

roc_score

f2_score <- fbeta(test$y, preds, positive="yes", beta = 2)
f2_score

confusionMatrix(data=preds, reference = test$y)

# Ridge
mod.out <- glmnet(x=as.matrix(subset(val_train_dummies_over_sc, select = -c(y))), 
                  y=as.matrix(as.numeric(val_train_dummies_over_sc$y)-1), 
                  family = binomial, alpha = 0)

summary(mod.out)
probabilities <- predict(mod.out, as.matrix(subset(val_val_dummies_sc, select = -c(y))), 
                         type="response")
roc_auc_scores <- sapply(as.data.frame(probabilities), function(x) roc(val_val_dummies_sc$y, x)$auc)
thresholds <- sapply(as.data.frame(probabilities), function(x) adjust_threshold(val_val_dummies_sc$y, x))

f2_scores <- c()
for (i in 1:length(thresholds)) {
  proba <- probabilities[, i]
  trsh <- thresholds[i]
  preds <- ifelse(proba > trsh, "yes", "no")
  preds <- factor(preds, levels=c("no", "yes"))
  f2_score <- fbeta(val_val_dummies_sc$y, preds, positive="yes", beta = 2)
  f2_scores <- c(f2_scores, f2_score)
}

lambdas <- mod.out$lambda
plot(lambdas, roc_auc_scores, type="l")
plot(lambdas, f2_scores, type="l")

#lambda = lambdas[roc_auc_scores == max(roc_auc_scores)]
lambda = lambdas[f2_scores == max(f2_scores)]
lambda
threshold = thresholds[f2_scores == max(f2_scores)]
threshold

mod.out <- glmnet(x=as.matrix(subset(train_dummies_over_sc, select = -c(y))), 
                  y=as.matrix(as.numeric(train_dummies_over_sc$y)-1), 
                  family = binomial, alpha = 0, lambda=lambda)

probabilities <- predict(mod.out, as.matrix(subset(test_dummies_sc,
                                                   select = -c(y))), type="response")
probabilities <- as.vector(probabilities)
preds <- ifelse(probabilities > threshold, "yes", "no")
preds <- as.factor(preds)

roc_score=roc(test$y, probabilities) #AUC score
plot(roc_score ,main ="ROC curve -- Logistic Regression ")

roc_score

f2_score <- fbeta(test$y, preds, positive="yes", beta = 2)
f2_score

confusionMatrix(data=preds, reference = test$y)

# Lasso
mod.out <- glmnet(x=as.matrix(subset(val_train_dummies_over_sc, select = -c(y))), 
                  y=as.matrix(as.numeric(val_train_dummies_over_sc$y)-1), 
                  family = binomial, alpha = 1)

summary(mod.out)
probabilities <- predict(mod.out, as.matrix(subset(val_val_dummies_sc, select = -c(y))), 
                         type="response")
roc_auc_scores <- sapply(as.data.frame(probabilities), function(x) roc(val_val_dummies_sc$y, x)$auc)
thresholds <- sapply(as.data.frame(probabilities), function(x) adjust_threshold(val_val_dummies_sc$y, x))

f2_scores <- c()
for (i in 1:length(thresholds)) {
  proba <- probabilities[, i]
  trsh <- thresholds[i]
  preds <- ifelse(proba > trsh, "yes", "no")
  preds <- factor(preds, levels=c("no", "yes"))
  f2_score <- fbeta(val_val_dummies_sc$y, preds, positive="yes", beta = 2)
  f2_scores <- c(f2_scores, f2_score)
}

lambdas <- mod.out$lambda
plot(lambdas, roc_auc_scores, type="l")
plot(lambdas, f2_scores, type="l")

#lambda = lambdas[roc_auc_scores == max(roc_auc_scores)]
lambda = lambdas[f2_scores == max(f2_scores)]
lambda
threshold = thresholds[f2_scores == max(f2_scores)]
threshold

mod.out <- glmnet(x=as.matrix(subset(train_dummies_over_sc, select = -c(y))), 
                  y=as.matrix(as.numeric(train_dummies_over_sc$y)-1), 
                  family = binomial, alpha = 1, lambda=lambda)

probabilities <- predict(mod.out, as.matrix(subset(test_dummies_sc,
                                                   select = -c(y))), type="response")
probabilities <- as.vector(probabilities)
preds <- ifelse(probabilities > threshold, "yes", "no")
preds <- as.factor(preds)

roc_score=roc(test$y, probabilities) #AUC score
plot(roc_score ,main ="ROC curve -- Logistic Regression ")

roc_score

f2_score <- fbeta(test$y, preds, positive="yes", beta = 2)
f2_score

confusionMatrix(data=preds, reference = test$y)


mod.out <- qda(y ~., data=train)
summary(mod.out)
probabilities <- predict(mod.out, test, type="response")
preds <- ifelse(probabilities > 0.5, "no", "yes")
preds <- as.factor(preds)

roc_score=roc(test$y, probabilities) #AUC score
plot(roc_score ,main ="ROC curve -- Logistic Regression ")

roc_score

confusionMatrix(data=preds, reference = test$y)

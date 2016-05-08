library(data.table)
library(lubridate)
library(Matrix)

#Linear Booster Model
dt <- fread("Train_seers_accuracy.csv")
dt <- dt[, YearTrans := str_sub(Transaction_Date, 7)]
dt <- dt[nchar(YearTrans) == 3, YearTrans := str_sub(YearTrans, 2)]
dt <- dt[, DOB := str_sub(DOB, 7)]
dt <- dt[nchar(DOB) == 3, DOB := str_sub(DOB, 2)]

dt[, Var1 := as.factor(Var1)]
dt[, Var2 := as.factor(Var2)]
dt[, Var3 := as.factor(Var3)]

train06 <- copy(dt)
train05 <- copy(dt)
train04 <- copy(dt)
clients06 <- train06[YearTrans == '06'][['Client_ID']]
clients05 <- train05[YearTrans == '05'][['Client_ID']]
clients04 <- train04[YearTrans == '04'][['Client_ID']]
train06 <- train06[YearTrans %in% c('03','04','05')][Client_ID %in% clients06, target := 1]
train05 <- train05[YearTrans %in% c('03','04')][Client_ID %in% clients05, target := 1]
train04 <- train04[YearTrans %in% c('03')][Client_ID %in% clients05, target := 1]
train06[is.na(target), target := 0]
train05[is.na(target), target := 0]
train04[is.na(target), target := 0]
clients_all <- dt[['Client_ID']]
## Add DOB

features <- c('Referred_Friend', 'Gender', 'Sales_Executive_Category', 'Payment_Mode',
              'Product_Category', 'Transaction_Amount', 'Var1', 'Var2', 'Var3',
              'Purchased_in_Sale', 'Number_of_EMI', 'Store_ID', 'YearTrans', 'Client_ID', 'DOB', 'Sales_Executive_ID')


train06 <- subset(train06, select = c(features, 'target'))
train05 <- subset(train05, select = c(features, 'target'))
train04 <- subset(train04, select = c(features, 'target'))
test <- subset(dt, select = features)

train06[, YearTrans := (6 - as.numeric(YearTrans))]
train05[, YearTrans := (5 - as.numeric(YearTrans))]
train04[, YearTrans := (4 - as.numeric(YearTrans))]
test[,  YearTrans := (7 - as.numeric(YearTrans))]

train06[, Age := 2006 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]
train05[, Age := 2005 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]
train04[, Age := 2004 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]
test[,  Age := 2007 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]

train06[, Freq := .N, by = Client_ID]
train05[, Freq := .N, by = Client_ID]
train04[, Freq := .N, by = Client_ID]
test[, Freq := .N, by = Client_ID]

train06[, FreqUniqStores := length(unique(Store_ID)), by = 'Client_ID']
train05[, FreqUniqStores := length(unique(Store_ID)), by = 'Client_ID']
train04[, FreqUniqStores := length(unique(Store_ID)), by = 'Client_ID']
test[,  FreqUniqStores := length(unique(Store_ID)), by = 'Client_ID']

train06[, FreqUniqEMI := length(unique(Number_of_EMI)), by = 'Client_ID']
train05[, FreqUniqEMI := length(unique(Number_of_EMI)), by = 'Client_ID']
train04[, FreqUniqEMI := length(unique(Number_of_EMI)), by = 'Client_ID']
test[,  FreqUniqEMI := length(unique(Number_of_EMI)), by = 'Client_ID']

train06[, NumCategory := length(unique(Product_Category)), by = 'Client_ID']
train05[, NumCategory := length(unique(Product_Category)), by = 'Client_ID']
train04[, NumCategory := length(unique(Product_Category)), by = 'Client_ID']
test[, NumCategory := length(unique(Product_Category)), by = 'Client_ID']

train06[, SameSales := .N, by = c('Client_ID', 'Sales_Executive_ID')]
train05[, SameSales := .N, by = c('Client_ID', 'Sales_Executive_ID')]
train04[, SameSales := .N, by = c('Client_ID', 'Sales_Executive_ID')]
test[, SameSales := .N, by = c('Client_ID', 'Sales_Executive_ID')]

clients_train06 <- train06[['Client_ID']]
clients_train05 <- train05[['Client_ID']]
clients_train04 <- train04[['Client_ID']]

returned_clients06 <- train06[target == 1][['Client_ID']]
returned_clients05 <- train05[target == 1][['Client_ID']]
returned_clients04 <- train04[target == 1][['Client_ID']]

returned_clients0506 <- intersect(returned_clients06, returned_clients05)
always_return_clients <- intersect(returned_clients0506, returned_clients04)

clients_test    <- test[['Client_ID']]

test[, target := -1]

total_train   <- rbindlist(list(train06, train05, train04))
clients_train <- c(clients_train06, clients_train05, clients_train04)

test[, target := -1]
total <- rbindlist(list(copy(total_train),
                        copy(test)),
                   use.names = TRUE)

total[, Number_of_EMI := as.factor(Number_of_EMI)]
total[, target_sparse := target]
total[, client_id_fake := 1]

full <- sparse.model.matrix(target_sparse ~ ., data = total)

s    <- colSums(full)
fil  <- (s < 50)
fil["target"] <- FALSE
full <- full[, -which(fil)]

train_sp06 <- full[1:nrow(train06), ]
train_sp05 <- full[(nrow(train06)+1):(nrow(train06) + nrow(train05)), ]
train_sp04 <- full[(nrow(train06) + nrow(train05) + 1):nrow(total_train), ]
test_sp  <- full[(nrow(total_train)+1):nrow(full), ]

aggregate_by <- function(x, group_by) { 
  x.T   <- as(x, "TsparseMatrix")
  x.T@i <- as.integer(as.integer(group_by[x.T@i+1])-1) 
  y     <- as(x.T, "CsparseMatrix")
  y@Dim[1] <- nlevels(group_by)
  y
} 

get.column.from.sparse.matrix <- function(matrix, column) {
  tr_nn <- matrix@Dimnames[[2]]
  cl <- which(tr_nn %in% column)
  res <- matrix[1:nrow(matrix), cl]
  return(res)
}

remove.column.from.sparse.matrix <- function(matrix, column) {
  tr_nn <- matrix@Dimnames[[2]]
  cl <- which(tr_nn %in% column)
  matrix <- matrix[, -cl]
  return(matrix)
}

train_sp06 <- aggregate_by(train_sp06, factor(train06[['Client_ID']]))
train_sp05 <- aggregate_by(train_sp05, factor(train05[['Client_ID']]))
train_sp04 <- aggregate_by(train_sp04, factor(train04[['Client_ID']]))
test_sp  <- aggregate_by(test_sp,  factor(test[['Client_ID']]))

train06_count <- get.column.from.sparse.matrix(train_sp06, 'client_id_fake')
train05_count <- get.column.from.sparse.matrix(train_sp05, 'client_id_fake')
train04_count <- get.column.from.sparse.matrix(train_sp04, 'client_id_fake')
test_count  <- get.column.from.sparse.matrix(test_sp, 'client_id_fake')

train06_clients <- get.column.from.sparse.matrix(train_sp06, 'Client_ID')
train05_clients <- get.column.from.sparse.matrix(train_sp05, 'Client_ID')
train04_clients <- get.column.from.sparse.matrix(train_sp04, 'Client_ID')
test_clients  <- get.column.from.sparse.matrix(test_sp, 'Client_ID')

train06_target  <- get.column.from.sparse.matrix(train_sp06, 'target')
train05_target  <- get.column.from.sparse.matrix(train_sp05, 'target')
train04_target  <- get.column.from.sparse.matrix(train_sp04, 'target')
test_target   <- get.column.from.sparse.matrix(test_sp, 'target')

train06_clients <- train06_clients / train06_count
train05_clients <- train05_clients / train05_count
train04_clients <- train04_clients / train04_count
test_clients <- test_clients / test_count

# Check
table(train06_clients %in% train06[['Client_ID']])
table(train05_clients %in% train05[['Client_ID']])
table(train04_clients %in% train04[['Client_ID']])
table(test_clients %in% test[['Client_ID']])


train_sp <- rBind(train_sp06, train_sp05, train_sp04)
train_sp <- remove.column.from.sparse.matrix(train_sp, c('Client_ID', 'target', 'client_id_fake'))
test_sp <- remove.column.from.sparse.matrix(test_sp, c('Client_ID', 'target', 'client_id_fake'))

train_target <- ifelse(c(train06_target, train05_target, train04_target) >= 1, 1, 0)

dtrain <- xgb.DMatrix(data = train_sp, label = train_target)
dtest <- xgb.DMatrix(data = test_sp)
param <- list(objective           = "binary:logistic", 
              booster             = "gblinear",
              eval_metric         = "auc",
              eta                 = 0.15,
              alpha = 0.01, # EXP
              lambda = 0.01, # EXP
              max_depth           = 7)

watchlist <- list(train = dtrain)

clf <- xgb.train(params  = param, 
                 data    = dtrain, 
                 nrounds = 300, 
                 verbose = 1,
                 nfold   = 4,
                 watchlist = watchlist)
# 125 iteration CV ~ 0.855815
# 125 iteration LB ~  0.86262
# 265 iter LB ~ 0.86249
# 265 iter CV ~ 0.857426
preds <- predict(clf, dtest)
subs <- data.table(Cross_Sell = preds, Client_ID = test_clients)
subs <- subs[, .(Cross_Sell = mean(Cross_Sell)), by = Client_ID]
subs <- subset(subs, select = c('Client_ID', 'Cross_Sell'))
subs[, Cross_Sell := round(Cross_Sell, 4)]
subs[, Client_ID := as.character(Client_ID)]
write.csv(subs, "linear_fix2.csv", quote = FALSE, row.names = FALSE)


#XGBOOST1
library(data.table)
library(lubridate)
library(Matrix)
library(caret)
library(xgboost)
library(stringr)
dt <- fread("Train_seers_accuracy.csv")
dt <- dt[, YearTrans := str_sub(Transaction_Date, 7)]
dt <- dt[nchar(YearTrans) == 3, YearTrans := str_sub(YearTrans, 2)]
dt <- dt[, DOB := str_sub(DOB, 7)]
dt <- dt[nchar(DOB) == 3, DOB := str_sub(DOB, 2)]

dt[, Var1 := as.factor(Var1)]
dt[, Var2 := as.factor(Var1)]
dt[, Var3 := as.factor(Var1)]

train <- copy(dt)
clients <- train[YearTrans == '06'][['Client_ID']]
train <- train[Client_ID %in% clients, target := 1]
train[is.na(target), target := 0]
clients_all <- dt[['Client_ID']]
## Add DOB
train <- subset(train[YearTrans != '06'], select = c('Referred_Friend', 'Gender', 'Sales_Executive_Category', 'Payment_Mode',
                                                     'Product_Category', 'Transaction_Amount', 'Var1', 'Var2', 'Var3',
                                                     'Purchased_in_Sale', 'Number_of_EMI', 'Store_ID', 'target', 'YearTrans',
                                                     'Client_ID', 'DOB'))
test <- subset(dt, select = c('Referred_Friend', 'Gender', 'Sales_Executive_Category', 'Payment_Mode',
                              'Product_Category', 'Transaction_Amount', 'Var1', 'Var2', 'Var3',
                              'Purchased_in_Sale', 'Number_of_EMI', 'Store_ID', 'YearTrans', 'Client_ID',
                              'DOB'))

train[, YearTrans := (6 - as.numeric(YearTrans))]
test[,  YearTrans := (7 - as.numeric(YearTrans))]

train[, Transaction_AmountMean := mean(Transaction_Amount), by = 'Client_ID']
test[,  Transaction_AmountMean := mean(Transaction_Amount), by = 'Client_ID']

train[, Age := 2006 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]
test[,  Age := 2007 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]

#train[, NumRefFriends := sum(Referred_Friend == "YES"), by = 'Client_ID']
#test[, NumRefFriends := sum(Referred_Friend == "YES"), by = 'Client_ID']

#train[, DOB := NULL]
#test[, DOB := NULL]

train[, Freq := .N, by = Client_ID]
test[, Freq := .N, by = Client_ID]

clients_train <- train[['Client_ID']]
clients_test <- test[['Client_ID']]

labels <- train[, .(Target = ifelse(sum(target) == 0, 0, 1)), by = c('Client_ID')]

test[, target := -1]
total <- rbindlist(list(copy(train)[, Client_ID := NULL],
                        copy(test)[, Client_ID := NULL]),
                   use.names = TRUE)

full <- sparse.model.matrix(target ~ ., data = total)

s    <- colSums(full)
fil  <- (s < 150)
full <- full[, -which(fil)]

train_sp <- full[1:nrow(train), ]
test_sp  <- full[(nrow(train)+1):nrow(full), ]

aggregate_by <- function(x, group_by) { 
  x.T   <- as(x, "TsparseMatrix")
  x.T@i <- as.integer(as.integer(group_by[x.T@i+1])-1) 
  y     <- as(x.T, "CsparseMatrix")
  y@Dim[1] <- nlevels(group_by)
  y
} 

#train_sp <- aggregate_by(train_sp, as.factor(train[['Client_ID']]))
#test_sp  <- aggregate_by(test_sp,  as.factor(test[['Client_ID']]))

#train <- subset(train, select = c("Client_ID", "target"))
#train <- train[, .(target = 1 * (sum(target) > 0)), by = 'Client_ID']

#setkey(train, Client_ID)
#train <- unique(train)
#target_train <- train[["target"]]

#test_clients <- unique(test$Client_ID)
split <- createFolds(as.factor(train$target),10)
score <- c()
for(i in 1:10){
  dtrain <- xgb.DMatrix(data = train_sp[-split[[i]],], label = train$target[-split[[i]]])
  dval <- xgb.DMatrix(data=train_sp[split[[i]],],label = train$target[split[[i]]])
  dtest <- xgb.DMatrix(data = test_sp)
  param <- list(objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.1,
                max_depth           = 7,
                lambda=0.001,
                alpha=1.5,
                gamma=0.5,
                nthread = 36)
  
  watchlist <- list(val=dval,train=dtrain)
  
  clf <- xgb.train(params  = param, 
                   data    = dtrain, 
                   nrounds = 500, 
                   verbose = 1,
                   watchlist = watchlist,
                   print.every.n = 20,
                   early.stop.round = 20)
  if(i==1) pred1 <- predict(clf,dtest) else pred1 <- pred1 + predict(clf,dtest)
  if(i==1) pred2 <- predict(clf,dtest)^2 else pred2 <- pred2 + predict(clf,dtest)^2
  print(i)
  score <- c(score,clf$bestScore)
  print(mean(score))
}

preds <- sqrt(pred2/10)
subs <- data.table(Cross_Sell = preds, Client_ID = dt$Client_ID)
subs <- subs[, .(Cross_Sell = mean(Cross_Sell)), by = Client_ID]
subs <- subset(subs, select = c('Client_ID', 'Cross_Sell'))
subs[, Cross_Sell := round(Cross_Sell, 4)]
write.csv(subs, "dataset1_10avg_xgb2.csv", quote = FALSE, row.names = FALSE)


#XGBOOST2
library(data.table)
library(lubridate)
library(Matrix)
library(caret)
library(xgboost)
library(stringr)

dt <- fread("Train_seers_accuracy.csv")
dt <- dt[, YearTrans := str_sub(Transaction_Date, 7)]
dt <- dt[nchar(YearTrans) == 3, YearTrans := str_sub(YearTrans, 2)]
dt <- dt[, DOB := str_sub(DOB, 7)]
dt <- dt[nchar(DOB) == 3, DOB := str_sub(DOB, 2)]

dt[, Var1 := as.factor(Var1)]
dt[, Var2 := as.factor(Var1)]
dt[, Var3 := as.factor(Var1)]

train <- copy(dt)
clients <- train[YearTrans == '06'][['Client_ID']]
train <- train[Client_ID %in% clients, target := 1]
train[is.na(target), target := 0]
clients_all <- dt[['Client_ID']]
## Add DOB
train <- subset(train[YearTrans != '06'], select = c('Referred_Friend', 'Gender', 'Sales_Executive_Category', 'Payment_Mode',
                                                     'Product_Category', 'Transaction_Amount', 'Var1', 'Var2', 'Var3',
                                                     'Purchased_in_Sale', 'Number_of_EMI', 'Store_ID', 'target', 'YearTrans',
                                                     'Client_ID', 'DOB'))
test <- subset(dt, select = c('Referred_Friend', 'Gender', 'Sales_Executive_Category', 'Payment_Mode',
                              'Product_Category', 'Transaction_Amount', 'Var1', 'Var2', 'Var3',
                              'Purchased_in_Sale', 'Number_of_EMI', 'Store_ID', 'YearTrans', 'Client_ID',
                              'DOB'))
train[, YearTrans := (6 - as.numeric(YearTrans))]
test[,  YearTrans := (7 - as.numeric(YearTrans))]

train[, Transaction_AmountMean := mean(Transaction_Amount), by = 'Client_ID']
test[,  Transaction_AmountMean := mean(Transaction_Amount), by = 'Client_ID']

train[, Age := 2006 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]
test[,  Age := 2007 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]

#train[, NumRefFriends := sum(Referred_Friend == "YES"), by = 'Client_ID']
#test[, NumRefFriends := sum(Referred_Friend == "YES"), by = 'Client_ID']

#train[, DOB := NULL]
#test[, DOB := NULL]

train[, Freq := .N, by = Client_ID]
test[, Freq := .N, by = Client_ID]

clients_train <- train[['Client_ID']]
clients_test <- test[['Client_ID']]

labels <- train[, .(Target = ifelse(sum(target) == 0, 0, 1)), by = c('Client_ID')]

test[, target := -1]
total <- rbindlist(list(copy(train)[, Client_ID := NULL],
                        copy(test)[, Client_ID := NULL]),
                   use.names = TRUE)
total <- total[, lapply(.SD, function(x) {as.numeric(as.factor(x))})]

full <- data.matrix(total[, target := NULL])

train_sp <- full[1:nrow(train), ]
test_sp  <- full[(nrow(train)+1):nrow(full), ]

split <- createFolds(as.factor(train$target),10)
score <- c()
for(i in 1:10){
  dtrain <- xgb.DMatrix(data = train_sp[-split[[i]],], label = train$target[-split[[i]]])
  dval <- xgb.DMatrix(data=train_sp[split[[i]],],label = train$target[split[[i]]])
  dtest <- xgb.DMatrix(data = test_sp)
  param <- list(objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.1,
                max_depth           = 7,
                lambda=0.001,
                alpha=1.5,
                gamma=0.5,
                nthread = 36)
  
  watchlist <- list(val=dval,train=dtrain)
  clf <- xgb.train(params  = param, 
                   data    = dtrain, 
                   nrounds = 500, 
                   verbose = 1,
                   watchlist = watchlist,
                   print.every.n = 20,
                   early.stop.round = 20)
  if(i==1) pred1 <- predict(clf,dtest) else pred1 <- pred1 + predict(clf,dtest)
  if(i==1) pred2 <- predict(clf,dtest)^2 else pred2 <- pred2 + predict(clf,dtest)^2
  print(i)
  score <- c(score,clf$bestScore)
  print(mean(score))
}

preds <- pred1/10
subs <- data.table(Cross_Sell = preds, Client_ID = dt$Client_ID)
subs <- subs[, .(Cross_Sell = mean(Cross_Sell)), by = Client_ID]
subs <- subset(subs, select = c('Client_ID', 'Cross_Sell'))
subs[, Cross_Sell := round(Cross_Sell, 4)]
write.csv(subs, "dataset2_10avg_xgb1.csv", quote = FALSE, row.names = FALSE)

#XGBOOST3
library(data.table)
library(lubridate)
library(Matrix)
library(caret)
library(xgboost)
library(stringr)

dt <- fread("Train_seers_accuracy.csv")
dt <- dt[, YearTrans := str_sub(Transaction_Date, 7)]
dt <- dt[nchar(YearTrans) == 3, YearTrans := str_sub(YearTrans, 2)]
dt <- dt[, DOB := str_sub(DOB, 7)]
dt <- dt[nchar(DOB) == 3, DOB := str_sub(DOB, 2)]

dt[, Var1 := as.factor(Var1)]
dt[, Var2 := as.factor(Var2)]
dt[, Var3 := as.factor(Var3)]

train <- copy(dt)
clients <- train[YearTrans == '06'][['Client_ID']]
train <- train[Client_ID %in% clients, target := 1]
train[is.na(target), target := 0]
clients_all <- dt[['Client_ID']]
## Add DOB
train <- subset(train[YearTrans != '06'], select = c('Referred_Friend', 'Gender', 'Sales_Executive_Category', 'Payment_Mode',
                                                     'Product_Category', 'Transaction_Amount', 'Var1', 'Var2', 'Var3',
                                                     'Purchased_in_Sale', 'Number_of_EMI', 'Store_ID', 'target', 'YearTrans',
                                                     'Client_ID', 'DOB'))
test <- subset(dt, select = c('Referred_Friend', 'Gender', 'Sales_Executive_Category', 'Payment_Mode',
                              'Product_Category', 'Transaction_Amount', 'Var1', 'Var2', 'Var3',
                              'Purchased_in_Sale', 'Number_of_EMI', 'Store_ID', 'YearTrans', 'Client_ID',
                              'DOB'))

train[, YearTrans := (6 - as.numeric(YearTrans))]
test[,  YearTrans := (7 - as.numeric(YearTrans))]

train[, Transaction_AmountMean := mean(Transaction_Amount), by = 'Client_ID']
test[,  Transaction_AmountMean := mean(Transaction_Amount), by = 'Client_ID']

train[, Age := 2006 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]
test[,  Age := 2007 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]

#train[, NumRefFriends := sum(Referred_Friend == "YES"), by = 'Client_ID']
#test[, NumRefFriends := sum(Referred_Friend == "YES"), by = 'Client_ID']

#train[, DOB := NULL]
#test[, DOB := NULL]

train[, Freq := .N, by = Client_ID]
test[, Freq := .N, by = Client_ID]

clients_train <- train[['Client_ID']]
clients_test <- test[['Client_ID']]

labels <- train[, .(Target = ifelse(sum(target) == 0, 0, 1)), by = c('Client_ID')]

test[, target := -1]
total <- rbindlist(list(copy(train),copy(test)),
                   use.names = TRUE)

total[, Number_of_EMI := as.factor(Number_of_EMI)]
total[, Freq := as.factor(Freq)]
total[, Transaction_Amount := as.factor(round(Transaction_Amount / 1000, 0))]
total[, target_sparse := target]
total[, client_id_fake := 1]
full <- sparse.model.matrix(target_sparse ~ ., data = total)

s    <- colSums(full)
fil  <- (s < 150)
fil["target"] <- FALSE
full <- full[, -which(fil)]

train_sp <- full[1:nrow(train), ]
test_sp  <- full[(nrow(train)+1):nrow(full), ]

aggregate_by <- function(x, group_by) { 
  x.T   <- as(x, "TsparseMatrix")
  x.T@i <- as.integer(as.integer(group_by[x.T@i+1])-1) 
  y     <- as(x.T, "CsparseMatrix")
  y@Dim[1] <- nlevels(group_by)
  y
} 

get.column.from.sparse.matrix <- function(matrix, column) {
  tr_nn <- matrix@Dimnames[[2]]
  cl <- which(tr_nn %in% column)
  res <- matrix[1:nrow(matrix), cl]
  return(res)
}

remove.column.from.sparse.matrix <- function(matrix, column) {
  tr_nn <- matrix@Dimnames[[2]]
  cl <- which(tr_nn %in% column)
  matrix <- matrix[, -cl]
  return(matrix)
}

train_sp <- aggregate_by(train_sp, factor(train[['Client_ID']]))
test_sp  <- aggregate_by(test_sp,  factor(test[['Client_ID']]))

train_count <- get.column.from.sparse.matrix(train_sp, 'client_id_fake')
test_count  <- get.column.from.sparse.matrix(test_sp, 'client_id_fake')
train_clients <- get.column.from.sparse.matrix(train_sp, 'Client_ID')
test_clients  <- get.column.from.sparse.matrix(test_sp, 'Client_ID')
train_target  <- get.column.from.sparse.matrix(train_sp, 'target')
test_target   <- get.column.from.sparse.matrix(test_sp, 'target')

train_clients <- train_clients / train_count
test_clients <- test_clients / test_count

# Check
table(train_clients %in% train[['Client_ID']])
table(test_clients %in% test[['Client_ID']])

train_sp <- remove.column.from.sparse.matrix(train_sp, c('Client_ID', 'target', 'client_id_fake'))
test_sp <- remove.column.from.sparse.matrix(test_sp, c('Client_ID', 'target', 'client_id_fake'))

train_target <- ifelse(train_target >= 1, 1, 0)

split <- createFolds(as.factor(train_target),10)

score <- c()
for(i in 1:10){
  dtrain <- xgb.DMatrix(data = train_sp[-split[[i]],], label = train_target[-split[[i]]])
  dval <- xgb.DMatrix(data=train_sp[split[[i]],],label = train_target[split[[i]]])
  dtest <- xgb.DMatrix(data = test_sp)
  param <- list(objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.1,
                max_depth           = 7,
                alpha = 1.5,
                lambda = 0.001,
                gamma = 0.3,
                nthread = 36)
  
  watchlist <- list(val=dval,train=dtrain)
  
  clf <- xgb.train(params  = param, 
                   data    = dtrain, 
                   nrounds = 500, 
                   verbose = 1,
                   watchlist = watchlist,
                   print.every.n = 20,
                   early.stop.round = 20)
  if(i==1) pred1 <- predict(clf,dtest) else pred1 <- pred1 + predict(clf,dtest)
  if(i==1) pred2 <- predict(clf,dtest)^2 else pred2 <- pred2 + predict(clf,dtest)^2
  print(i)
  score <- c(score,clf$bestScore)
  print(mean(score))
}


preds <- sqrt(pred2/10)
subs <- data.table(Cross_Sell = preds, Client_ID = test_clients)
subs <- subs[, .(Cross_Sell = mean(Cross_Sell)), by = Client_ID]
subs <- subset(subs, select = c('Client_ID', 'Cross_Sell'))
subs[, Cross_Sell := round(Cross_Sell, 4)]
subs[, Client_ID := as.character(Client_ID)]

write.csv(subs, "dataset3_10avg_xgb2.csv", quote = FALSE, row.names = FALSE)

#XGBOOST4
library(data.table)
library(lubridate)
library(Matrix)
library(caret)
library(xgboost)
library(stringr)
library(reshape2)

sparse2triples <- function(m) {
  SM = summary(m)
  D1 = m@Dimnames[[1]][SM[,1]]
  D2 = m@Dimnames[[2]][SM[,2]]
  data.frame(row=D1, col=D2, x=m@x)
}

dt <- fread("Train_seers_accuracy.csv")
dt <- dt[, YearTrans := str_sub(Transaction_Date, 7)]
dt <- dt[nchar(YearTrans) == 3, YearTrans := str_sub(YearTrans, 2)]
dt <- dt[, DOB := str_sub(DOB, 7)]
dt <- dt[nchar(DOB) == 3, DOB := str_sub(DOB, 2)]

dt[, Var1 := as.factor(Var1)]
dt[, Var2 := as.factor(Var2)]
dt[, Var3 := as.factor(Var3)]

train06 <- copy(dt)
train05 <- copy(dt)
train04 <- copy(dt)
clients06 <- train06[YearTrans == '06'][['Client_ID']]
clients05 <- train05[YearTrans == '05'][['Client_ID']]
clients04 <- train04[YearTrans == '04'][['Client_ID']]
train06 <- train06[YearTrans %in% c('03','04','05')][Client_ID %in% clients06, target := 1]
train05 <- train05[YearTrans %in% c('03','04')][Client_ID %in% clients05, target := 1]
train04 <- train04[YearTrans %in% c('03')][Client_ID %in% clients05, target := 1]
train06[is.na(target), target := 0]
train05[is.na(target), target := 0]
train04[is.na(target), target := 0]
clients_all <- dt[['Client_ID']]
## Add DOB

features <- c('Referred_Friend', 'Gender', 'Sales_Executive_Category', 'Payment_Mode',
              'Product_Category', 'Transaction_Amount', 'Var1', 'Var2', 'Var3',
              'Purchased_in_Sale', 'Number_of_EMI', 'Store_ID', 'YearTrans', 'Client_ID', 'DOB', 'Sales_Executive_ID')


train06 <- subset(train06, select = c(features, 'target'))
train05 <- subset(train05, select = c(features, 'target'))
train04 <- subset(train04, select = c(features, 'target'))
test <- subset(dt, select = features)

train06[, YearTrans := (6 - as.numeric(YearTrans))]
train05[, YearTrans := (5 - as.numeric(YearTrans))]
train04[, YearTrans := (4 - as.numeric(YearTrans))]
test[,  YearTrans := (7 - as.numeric(YearTrans))]

train06[, Age := 2006 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]
train05[, Age := 2005 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]
train04[, Age := 2004 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]
test[,  Age := 2007 - as.numeric(paste0('19', DOB[1], 'collapse' = '')), by = 'Client_ID'][, DOB := NULL]

train06[, Freq := .N, by = Client_ID]
train05[, Freq := .N, by = Client_ID]
train04[, Freq := .N, by = Client_ID]
test[, Freq := .N, by = Client_ID]

train06[, FreqUniqStores := length(unique(Store_ID)), by = 'Client_ID']
train05[, FreqUniqStores := length(unique(Store_ID)), by = 'Client_ID']
train04[, FreqUniqStores := length(unique(Store_ID)), by = 'Client_ID']
test[,  FreqUniqStores := length(unique(Store_ID)), by = 'Client_ID']

train06[, FreqUniqEMI := length(unique(Number_of_EMI)), by = 'Client_ID']
train05[, FreqUniqEMI := length(unique(Number_of_EMI)), by = 'Client_ID']
train04[, FreqUniqEMI := length(unique(Number_of_EMI)), by = 'Client_ID']
test[,  FreqUniqEMI := length(unique(Number_of_EMI)), by = 'Client_ID']

train06[, NumCategory := length(unique(Product_Category)), by = 'Client_ID']
train05[, NumCategory := length(unique(Product_Category)), by = 'Client_ID']
train04[, NumCategory := length(unique(Product_Category)), by = 'Client_ID']
test[, NumCategory := length(unique(Product_Category)), by = 'Client_ID']

train06[, SameSales := .N, by = c('Client_ID', 'Sales_Executive_ID')]
train05[, SameSales := .N, by = c('Client_ID', 'Sales_Executive_ID')]
train04[, SameSales := .N, by = c('Client_ID', 'Sales_Executive_ID')]
test[, SameSales := .N, by = c('Client_ID', 'Sales_Executive_ID')]

clients_train06 <- train06[['Client_ID']]
clients_train05 <- train05[['Client_ID']]
clients_train04 <- train04[['Client_ID']]

returned_clients06 <- train06[target == 1][['Client_ID']]
returned_clients05 <- train05[target == 1][['Client_ID']]
returned_clients04 <- train04[target == 1][['Client_ID']]

returned_clients0506 <- intersect(returned_clients06, returned_clients05)
always_return_clients <- intersect(returned_clients0506, returned_clients04)

clients_test    <- test[['Client_ID']]

test[, target := -1]

total_train   <- rbindlist(list(train06, train05, train04))
clients_train <- c(clients_train06, clients_train05, clients_train04)

test[, target := -1]
total <- rbindlist(list(copy(total_train),
                        copy(test)),
                   use.names = TRUE)

total[, Number_of_EMI := as.factor(Number_of_EMI)]
total[, target_sparse := target]
total[, client_id_fake := 1]

full <- sparse.model.matrix(target_sparse ~ ., data = total)

s    <- colSums(full)
fil  <- (s < 150)
fil["target"] <- FALSE
full <- full[, -which(fil)]

train_sp06 <- full[1:nrow(train06), ]
train_sp05 <- full[(nrow(train06)+1):(nrow(train06) + nrow(train05)), ]
train_sp04 <- full[(nrow(train06) + nrow(train05) + 1):nrow(total_train), ]
test_sp  <- full[(nrow(total_train)+1):nrow(full), ]

aggregate_by <- function(x, group_by) { 
  x.T   <- as(x, "TsparseMatrix")
  x.T@i <- as.integer(as.integer(group_by[x.T@i+1])-1) 
  y     <- as(x.T, "CsparseMatrix")
  y@Dim[1] <- nlevels(group_by)
  y
} 

get.column.from.sparse.matrix <- function(matrix, column) {
  tr_nn <- matrix@Dimnames[[2]]
  cl <- which(tr_nn %in% column)
  res <- matrix[1:nrow(matrix), cl]
  return(res)
}

remove.column.from.sparse.matrix <- function(matrix, column) {
  tr_nn <- matrix@Dimnames[[2]]
  cl <- which(tr_nn %in% column)
  matrix <- matrix[, -cl]
  return(matrix)
}

train_sp06 <- aggregate_by(train_sp06, factor(train06[['Client_ID']]))
train_sp05 <- aggregate_by(train_sp05, factor(train05[['Client_ID']]))
train_sp04 <- aggregate_by(train_sp04, factor(train04[['Client_ID']]))
test_sp  <- aggregate_by(test_sp,  factor(test[['Client_ID']]))

train06_count <- get.column.from.sparse.matrix(train_sp06, 'client_id_fake')
train05_count <- get.column.from.sparse.matrix(train_sp05, 'client_id_fake')
train04_count <- get.column.from.sparse.matrix(train_sp04, 'client_id_fake')
test_count  <- get.column.from.sparse.matrix(test_sp, 'client_id_fake')

train06_clients <- get.column.from.sparse.matrix(train_sp06, 'Client_ID')
train05_clients <- get.column.from.sparse.matrix(train_sp05, 'Client_ID')
train04_clients <- get.column.from.sparse.matrix(train_sp04, 'Client_ID')
test_clients  <- get.column.from.sparse.matrix(test_sp, 'Client_ID')

train06_target  <- get.column.from.sparse.matrix(train_sp06, 'target')
train05_target  <- get.column.from.sparse.matrix(train_sp05, 'target')
train04_target  <- get.column.from.sparse.matrix(train_sp04, 'target')
test_target   <- get.column.from.sparse.matrix(test_sp, 'target')

train06_clients <- train06_clients / train06_count
train05_clients <- train05_clients / train05_count
train04_clients <- train04_clients / train04_count
test_clients <- test_clients / test_count

# Check
table(train06_clients %in% train06[['Client_ID']])
table(train05_clients %in% train05[['Client_ID']])
table(train04_clients %in% train04[['Client_ID']])
table(test_clients %in% test[['Client_ID']])


train_sp <- rBind(train_sp06, train_sp05, train_sp04)
train_sp <- remove.column.from.sparse.matrix(train_sp, c('Client_ID', 'target', 'client_id_fake'))
test_sp <- remove.column.from.sparse.matrix(test_sp, c('Client_ID', 'target', 'client_id_fake'))

train_target <- ifelse(c(train06_target, train05_target, train04_target) >= 1, 1, 0)

split <- createFolds(as.factor(train_target),10)

score <- c()
for(i in 1:10){
  dtrain <- xgb.DMatrix(data = train_sp[-split[[i]],], label = train_target[-split[[i]]])
  dval <- xgb.DMatrix(data=train_sp[split[[i]],],label = train_target[split[[i]]])
  dtest <- xgb.DMatrix(data = test_sp)
  param <- list(objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.1,
                max_depth           = 7,
                alpha = 1.5,
                lambda = 0.001,
                gamma = 0.3,
                nthread = 36)
  
  watchlist <- list(val=dval,train=dtrain)
  
  clf <- xgb.train(params  = param, 
                   data    = dtrain, 
                   nrounds = 500, 
                   verbose = 1,
                   watchlist = watchlist,
                   print.every.n = 20,
                   early.stop.round = 20)
  if(i==1) pred1 <- predict(clf,dtest) else pred1 <- pred1 + predict(clf,dtest)
  if(i==1) pred2 <- predict(clf,dtest)^2 else pred2 <- pred2 + predict(clf,dtest)^2
  print(i)
  score <- c(score,clf$bestScore)
  print(mean(score))
}


preds <- pred1/10
subs <- data.table(Cross_Sell = preds, Client_ID = test_clients)
subs <- subs[, .(Cross_Sell = mean(Cross_Sell)), by = Client_ID]
subs <- subset(subs, select = c('Client_ID', 'Cross_Sell'))
subs[, Cross_Sell := round(Cross_Sell, 4)]
subs[, Client_ID := as.character(Client_ID)]

write.csv(subs, "dataset4_10avg_xgb1.csv", quote = FALSE, row.names = FALSE)


xgb1 <- read.csv("linear_fix2.csv")
xgb2 <- read.csv("dataset1_10avg_xgb2.csv")
xgb3 <- read.csv("dataset2_10avg_xgb1.csv")
xgb4 <- read.csv("dataset3_10avg_xgb2.csv")
xgb5 <- read.csv("dataset4_10avg_xgb1.csv")

xgb1  <- xgb1[order(xgb1$Client_ID),]
xgb2  <- xgb2[order(xgb2$Client_ID),]
xgb3  <- xgb3[order(xgb3$Client_ID),]
xgb4  <- xgb4[order(xgb4$Client_ID),]
xgb5  <- xgb5[order(xgb5$Client_ID),]

ensemble1 <- xgb2
ensemble1[,2] <- sqrt((xgb2[,2]^2 + xgb3[,2]^2 + xgb4[,2]^2 + xgb5[,2]^2)/4)
ensemble <- xgb1
ensemble[,2] <- sqrt((xgb1[,2]^2 + ensemble[,2]^2)/2)

write.csv(ensemble,file="finalModel.csv",row.names=FALSE)

rm(list = ls())

library(MASS)
library(glmnet)
library(lbfgs) 
#library(iterators)
#library(doParallel)
library(cancerclass)

#details:
#1. use cv.glmnet to select the optimal tuning parameter lambda
#2. then calculate 4 types weights
#3. do the minmize problem

#get logit value
pi_beta <- function(beta, X){
  # X is the observation independent matrix
  Xbeta <- X %*% beta
  Xbeta[Xbeta>100] <- 100
  pibeta <- exp(Xbeta) / (1 + exp(Xbeta))
  return(pibeta)
}

#log-likelihood(scalar)
log_likelihood <- function(par, X, y, w) {
  #change into unweighted penalty
  #X_new <- X
  #for(s in 1:ncol(X)){
  #  X_new[, s] <- X[, s] / w[s]
  #}
  par <- par / w
  Xbeta <- X %*% par
  -(sum(y * Xbeta - log(1 + exp(Xbeta))))
}

#likelihood gradient(vector)
gradient <- function(par, X, y, w) {
  #X_new <- X
  #for(s in 1:ncol(X)){
  #  X_new[, s] <- X[, s] / w[s]
  #}
  par <- par / w
  p <- 1 / (1 + exp(- X %*% par))
  -(crossprod(X, (y - p)))
}

WLOG <- function(X, y, w, lambda = 1.0) {
  init <- rep(0, ncol(X))
  out_weighted <- lbfgs(log_likelihood, gradient, init, 
                        X = X, y = y, w = w,
                        orthantwise_c = lambda,
                        invisible = 1, linesearch_algorithm = "LBFGS_LINESEARCH_BACKTRACKING",
                        orthantwise_start = 0, orthantwise_end = ncol(X))
  #l1-penalty
  beta_new <- out_weighted$par
  conv <- out_weighted$convergence
  return(list(beta = beta_new / w, conv = conv))
}

## for application
w3_WLOG <- function(X, y, w3, lambda=1.0){
  X_new <- X
  for(s in 1:ncol(X)){
    X_new[, s] <- X[, s] / w3[s]
  }
  m <- glmnet(X_new, y, lambda = lambda, alpha = 1)
  beta <- m$beta[,1] / w3
  return(beta)
}

cal_weight_1 <- function(X, y, r) {
  n <- nrow(X)
  p <- ncol(X)
  cmax <- apply(abs(X), 2, max)
  w <- cmax * sqrt(2 / n * (r * log(p) + log(2)))
  w <- p * w / sum(w)
  w
}

cal_weight_2 <- function(X, y, r) {
  n <- nrow(X)
  p <- ncol(X)
  c_x <- sqrt(1 / n * apply(X^2, 2, sum))
  w <- c_x * sqrt(2 / n * (r * log(p) + log(2)))
  w <- p * w / sum(w)
  w
}

cal_weight_3 <- function(X, y, r) {
  n <- nrow(X)
  p <- ncol(X)
  X_colmean <- apply(X, 2, mean)
  X_new <- X
  for(i in 1:p){
    X_new[,i] <- X[,i]-X_colmean[i]
  }
  w <- 1 / sqrt(apply((X_new)^2, 2, sum) / n)
  w <- p * w / sum(w)
  w
}

cal_weight_4 <- function(X, y, lambda){
  model <- glmnet(X, y, family = "binomial", lambda = lambda, alpha = 1)
  beta <- model$beta[,1]
  #对当前估计值取绝对值倒数
  w <- rep(0, length(beta))
  for(i in 1:length(beta)){
    if(beta[i] == 0){
      w[i] <- 1/0.0001 
    } else {
      w[i] <- 1/abs(beta[i])
    }
  }
  w <- length(beta) * w / sum(w)
  return(w)
}

cal_mse_1 <- function(X_test, y_test, beta_est) {
  y_pred <- exp(X_test %*% beta_est)
  mse <- sum((y_pred - y_test)^2)
  mse
}
# the prediction error in table

cal_mse_2 <- function(X_test, beta_est, beta_true) {
  mu_true <- X_test %*% beta_true
  mu_est <- X_test %*% beta_est
  mse <- sqrt(sum((mu_true - mu_est)^2))
  mse
}

cal_mis <- function(X_test, beta_est, y_true){
  #计算模型的误分类率
  prob <- pi_beta(beta_est, X_test)
  y_est <- ifelse(prob > 0.5, 1, 0) 
  error <- sum(y_est!=y_true) / length(y_true)
  return(error)
}

#################### simulation 1--table 1
f1 <- function(N){
  Out_cv <- matrix(0, nrow = N, ncol = 10)
  
  for(i in 1:N){
    
    X <- mvrnorm(n = n_train, rep(0, p), Sigma)
    X_test <- mvrnorm(n = n_test, rep(0, p), Sigma)
    
    prob <- pi_beta(beta_true, X) 
    y <- rbinom(n = length(prob), size = 1, prob = prob)
    #use binomial distribution to generate y
    
    prob_test <- pi_beta(beta_true, X_test)
    y_test <- rbinom(n = length(prob_test), size = 1, prob = prob_test)
    
    init <- rep(0, ncol(X))
    
    X_std <- scale(X) / sqrt(n_train - 1) * sqrt(n_train)
    X_test_std <- scale(X_test) / sqrt(n_test - 1) * sqrt(n_test)
    
    ##use cv.glmnet to select best lambda
    m3 <- cv.glmnet(X_std, y, family = "binomial", alpha = 1)
    #nfold = 10 (default)
    cat('iteration: ', i, '\n')
    lam_cv <- m3$lambda.min
    cat('optimal lambda by cv of glmnet: ', lam_cv, '\n')
    
    ## use glmnet to find beta_lasso under the lam_cv. 
    m2_cv <- glmnet(X_std, y, lambda = lam_cv, family = "binomial", alpha = 1)
    beta_lasso_cv <- m2_cv$beta[,1]
    #cat('estimated beta by glmnet: ', beta_lasso_cv, '\n')
    l1_lasso_cv <- sum(abs(beta_lasso_cv - beta_true))
    #l1_lasso_nonzero_cv <- sum(abs(beta_lasso_cv[1:9] - beta_true[1:9]))
    mse_lasso_cv <- cal_mse_2(X_test, beta_lasso_cv, beta_true)
    
    ## weight 1
    w1 <- cal_weight_1(X, y, r)
    out1_cv <- WLOG(X, y, w1, lam_cv)
    beta1_cv <- out1_cv$beta 
    l1_w1_cv <- sum(abs(beta1_cv - beta_true))
    #l1_w1_nonzero_cv <- sum(abs(beta1_cv[1:9] - beta_true[1:9]))
    mse_w1_cv <- cal_mse_2(X_test, beta1_cv, beta_true)
  
    ## weight 2
    w2 <- cal_weight_2(X, y, r)
    #beta2_cv <- w2_WLOG(X, y, w2, lam_cv)
    out2_cv <- WLOG(X, y, w2, lam_cv)
    beta2_cv <- out2_cv$beta 
    l1_w2_cv <- sum(abs(beta2_cv - beta_true))
    #l1_w2_nonzero_cv <- sum(abs(beta2_cv[1:9]-beta_true[1:9]))
    mse_w2_cv <- cal_mse_2(X_test, beta2_cv, beta_true)

    ## weight 3
    w3 <- cal_weight_3(X, y, r)
    out3_cv <- WLOG(X, y, w3, lam_cv)
    beta3_cv <- out3_cv$beta 
    l1_w3_cv <- sum(abs(beta3_cv - beta_true))
    #l1_w3_nonzero_cv <- sum(abs(beta3_cv[1:9]-beta_true[1:9]))
    mse_w3_cv <- cal_mse_2(X_test, beta3_cv, beta_true)
    
    ##weight 4
    w4 <- cal_weight_4(X, y, lam_cv)
    beta4_cv <- WLOG(X, y, w4, lam_cv)$beta
    l1_w4_cv <- sum(abs(beta4_cv - beta_true))
    #l1_w4_nonzero_cv <- sum(abs(beta4_cv[1:9]-beta_true[1:9]))
    mse_w4_cv <- cal_mse_2(X_test, beta4_cv, beta_true)
    
    Out_cv[i,] <- c(l1_lasso_cv, l1_w1_cv, l1_w2_cv, l1_w3_cv, l1_w4_cv,
                    mse_lasso_cv, mse_w1_cv, mse_w2_cv, mse_w3_cv, mse_w4_cv)
#                    l1_lasso_nonzero_cv, l1_w1_nonzero_cv, l1_w2_nonzero_cv, l1_w3_nonzero_cv, l1_w4_nonzero_cv)
    }
  #Out = list(cv=Out_cv, bic=Out_bic, l1=Out_l1)
  #Out = list(cv=Out_cv, bic=Out_bic, diff=diff)
  #return(Out)
  return(Out_cv)
}

r <- 1
n_train <- 100
#n=100, p=25,50,75,100,125
n_test <- 200
p <- 200

#beta_true <- c(rep(10,9),rep(0, p-9))
#change the nonzero error calculation function in f1 function
beta_true <- c(rep(17, 3), rep(-5, 3), rep(13, 3), rep(0, p - 9))

# X's parameters
pho <- 0.5
Sigma <- array(0, dim = c(p, p))
for (i in 1:p) {
  for (j in 1:p) {
    Sigma[i, j] = pho^(abs(i - j))
  }
}

N <- 30
# number of simulation
Out <- f1(N)
#Out_cv <- Out$cv
#Out_bic <- Out$bic
#Out_l1 <- Out$l1

res_cv <- colMeans(Out)
#res_bic <- colMeans(Out_bic)
#res_l1 <- colMeans(Out_l1)

names(res_cv) <-c("l1_lasso_cv", "l1_w1_cv", "l1_w2_cv", "l1_w3_cv", "l1_w4_cv",
                  "mse_lasso_cv", "mse_w1_cv", "mse_w2_cv", "mse_w3_cv", "mse_w4_cv")
#                   "l1_lasso_nonzero", "l1_w1_nonzero", "l1_w2_nonzero", "l1_w3_nonzero")

res_cv


##################### application
###dataset 1
{
  data(GOLUB)
  data <- GOLUB@assayData[["exprs"]]
  y <- as.matrix(GOLUB@phenoData@data[,1], ncol = 1)
  
  X <- t(as.matrix(data))
  y[y=="ALL"] <- 1
  #acute lymphoblastic leukemia
  y[y=="AML"] <- 0
  #acute myeloid leukemia
  y <- as.numeric(y) 
  
  p <- ncol(X)
  n_train <- nrow(X)
  X_std <- scale(X) / sqrt(n_train - 1) * sqrt(n_train)
  r=1
  
  ##1. one time
  ##repeat cv.glmnet 200 times to find the average of cv_lambda
  lambda <- c()
  for(j in 1:200){
    m3 <- cv.glmnet(X_std, y, family = "binomial", alpha = 1)
    #nfold = 10 (default)
    lambda <- c(lambda, m3$lambda.min)
    cat(j,'\n')
  }
  lam_cv <- mean(lambda)
  cat('optimal lambda by cv of glmnet: ', lam_cv, '\n')
  
  lam <- lam_cv
  
  ## use glmnet to find beta_lasso under the lam_cv. 
  m2_cv <- glmnet(X_std, y, lambda = lam, family = "binomial", alpha = 1)
  beta_lasso_cv <- m2_cv$beta[,1]
  
  ## weight 1
  w1 <- cal_weight_1(X, y, r)
  beta1_cv <- WLOG(X, y, w1, lam)$beta
  
  ## weight 2
  w2 <- cal_weight_2(X, y, r)
  beta2_cv <- WLOG(X, y, w2, lam)$beta 
  
  ## weight 3
  w3 <- cal_weight_3(X, y, r)
  beta3_cv <-w3_WLOG(X, y, w3, lam)
  #beta3_cv <- WLOG(X, y, w3, lam)$beta 
  #use this function get all-zero estimation
  
  ##weight 4
  w4 <- cal_weight_4(X, y, lam)
  beta4_cv <- WLOG(X, y, w4, lam)$beta
  
  beta <- data.frame(lasso=beta_lasso_cv, w1=beta1_cv,
                     w2=beta2_cv, w3=beta3_cv, w4=beta4_cv)
  
  nonzero <- list(lasso = which(beta_lasso_cv != 0),
                  w1 = which(beta1_cv > 10^{-4}),
                  w2 = which(beta2_cv > 10^{-4}),
                  w3 = which(beta3_cv > 10^{-5}),
                  w4 = which(beta4_cv > 10^{-4}))
  lapply(nonzero, length)
  
  which(nonzero$lasso %in% nonzero$w1)
  which(nonzero$lasso %in% nonzero$w2)
  which(nonzero$lasso %in% nonzero$w3)
  
  zero <- list(w1 = which(beta$w1 <= 10^{-4}),
               w2 = which(beta$w2 <= 10^{-4}),
               w3 = which(beta$w3 <= 10^{-5}),
               w4 = which(beta$w4 <= 10^{-4}))
  
  beta$w1[zero$w1]=0
  beta$w2[zero$w2]=0
  beta$w3[zero$w3]=0
  beta$w4[zero$w4]=0
  
  ##error rate
  error_lasso <- cal_mis(X, beta_lasso_cv, y)
  error1 <- cal_mis(X, beta1_cv, y)
  error2 <- cal_mis(X, beta2_cv, y)
  error3 <- cal_mis(X, beta3_cv, y)
  error4 <- cal_mis(X, beta4_cv, y)
  
  error_onetime <- c(error_lasso,error1,error2,error3,error4)
  error_onetime
  
  ##show gene symbol
  gene <- colnames(X)
  gene_out <- list(lasso=gene[nonzero$lasso],
                   w1=gene[nonzero$w1],
                   w2=gene[nonzero$w2],
                   w3=gene[nonzero$w3],
                   w4=gene[nonzero$w4])
  write.csv(gene_out$lasso, file="~/Desktop/paper_Zhang/outcome/newoutcome/1_gene_out_lasso.csv")
  write.csv(gene_out$w1, file="~/Desktop/paper_Zhang/outcome/newoutcome/1_gene_out_w1.csv")
  write.csv(gene_out$w2, file="~/Desktop/paper_Zhang/outcome/newoutcome/1_gene_out_w2.csv")
  write.csv(gene_out$w3, file="~/Desktop/paper_Zhang/outcome/newoutcome/1_gene_out_w3.csv")
  write.csv(gene_out$w4, file="~/Desktop/paper_Zhang/outcome/newoutcome/1_gene_out_w4.csv")
  
  
  ##2. leave-one-out error rate
  modelsize <- matrix(0, nrow=nrow(X),ncol=5)
  error_matrix <- matrix(0, nrow=nrow(X), ncol=5)
  for(k in 1:nrow(X)){
    X_loo <- X[-k,]  
    y_loo <- y[-k]
    
    p <- ncol(X_loo)
    n_train <- nrow(X_loo)
    X_std <- scale(X_loo) / sqrt(n_train - 1) * sqrt(n_train)
    r=1
    
    ##use cv.glmnet to select best lambda
    m3 <- cv.glmnet(X_std, y_loo, family = "binomial", alpha = 1)
    #nfold = 10 (default)
    lam_cv <- m3$lambda.min
    cat('iteration', k, '\n')
    cat('optimal lambda by cv of glmnet: ', lam_cv, '\n')
    
    lam <- lam_cv
    
    ## use glmnet to find beta_lasso under the lam_cv. 
    m2_cv <- glmnet(X_std, y_loo, lambda = lam, family = "binomial", alpha = 1)
    beta_lasso_cv <- m2_cv$beta[,1]
    
    ## weight 1
    w1 <- cal_weight_1(X_loo, y_loo, r)
    beta1_cv <- WLOG(X_loo, y_loo, w1, lam)$beta
    
    ## weight 2
    w2 <- cal_weight_2(X_loo, y_loo, r)
    beta2_cv <- WLOG(X_loo, y_loo, w2, lam)$beta 
    
    ## weight 3
    w3 <- cal_weight_3(X_loo, y_loo, r)
    beta3_cv <- w3_WLOG(X_loo, y_loo, w3, lam)
    
    ##weight 4
    w4 <- cal_weight_4(X_loo, y_loo, lam)
    beta4_cv <- WLOG(X_loo, y_loo, w4, lam)$beta
    
    zero <- list(lasso = which(beta_lasso_cv == 0),
                 w1 = which(beta1_cv <= 1*10^{-5}),
                 w2 = which(beta2_cv <= 1*10^{-5}),
                 w3 = which(beta3_cv <= 1*10^{-5}),
                 w4 = which(beta4_cv <= 1*10^{-5}))
    ##modelsize
    modelsize[k,1] <- ncol(X)-length(zero$lasso)
    modelsize[k,2] <- ncol(X)-length(zero$w1)
    modelsize[k,3] <- ncol(X)-length(zero$w2)
    modelsize[k,4] <- ncol(X)-length(zero$w3)
    modelsize[k,5] <- ncol(X)-length(zero$w4)
    
    beta1_cv[zero$w1]=0
    beta2_cv[zero$w2]=0
    beta3_cv[zero$w3]=0
    beta4_cv[zero$w4]=0
    
    ##error rate
    error_matrix[k,1]  <- cal_mis(X[k,], beta_lasso_cv, y[k])
    error_matrix[k,2] <- cal_mis(X[k,], beta1_cv, y[k])
    error_matrix[k,3] <- cal_mis(X[k,], beta2_cv, y[k])
    error_matrix[k,4] <- cal_mis(X[k,], beta3_cv, y[k])
    error_matrix[k,5] <- cal_mis(X[k,], beta4_cv, y[k])
  }
  
  mean_ms <- apply(modelsize,2,mean)
  mean_error <- apply(error_matrix,2,mean)
  sd_ms <- apply(modelsize,2,sd)
  sd_error <- apply(error_matrix,2,sd)
  
  names(mean_ms) <- c("lasso","w1","w2","w3","w4")
  names(mean_error) <- c("lasso","w1","w2","w3","w4")
  names(sd_ms) <- c("lasso","w1","w2","w3","w4")
  names(sd_error) <- c("lasso","w1","w2","w3","w4")
  
  mean_ms
  mean_error
  sd_ms
  sd_error
  
}


##################### application
###dataset 2
{
  
  data(GOLUB1)
  data <- GOLUB1@assayData[["exprs"]]
  y <- as.matrix(GOLUB1@phenoData@data[,1], ncol = 1)
  X <- t(as.matrix(data))
  y[y=="ALL"] <- 1
  #acute lymphoblastic leukemia
  y[y=="AML"] <- 0
  #acute myeloid leukemia
  y <- as.numeric(y) 
  
  p <- ncol(X)
  n_train <- nrow(X)
  X_std <- scale(X) / sqrt(n_train - 1) * sqrt(n_train)
  r=1
  
  ##1.one time
  ##repeat cv.glmnet 200 times to find the average of cv_lambda
  lambda <- c()
  for(j in 1:200){
    m3 <- cv.glmnet(X_std, y, family = "binomial", alpha = 1)
    #nfold = 10 (default)
    lambda <- c(lambda, m3$lambda.min)
    cat(j,'\n')
  }
  lam_cv <- mean(lambda)
  cat('optimal lambda by cv of glmnet: ', lam_cv, '\n')
  
  lam <- lam_cv
  
  ## use glmnet to find beta_lasso under the lam_cv. 
  m2_cv <- glmnet(X_std, y, lambda = lam, family = "binomial", alpha = 1)
  beta_lasso_cv <- m2_cv$beta[,1]
  
  ## weight 1
  w1 <- cal_weight_1(X, y, r)
  beta1_cv <- WLOG(X, y, w1, lam)$beta
  
  ## weight 2
  w2 <- cal_weight_2(X, y, r)
  beta2_cv <- WLOG(X, y, w2, lam)$beta 
  
  ## weight 3
  w3 <- cal_weight_3(X, y, r)
  beta3_cv <- w3_WLOG(X, y, w3, lam)
  #beta3_cv <- WLOG(X, y, w3, lam)$beta 
  #use this function get all-zero estimation
  
  ##weight 4
  w4 <- cal_weight_4(X, y, lam)
  beta4_cv <- WLOG(X, y, w4, lam)$beta
  
  beta <- data.frame(lasso=beta_lasso_cv, w1=beta1_cv,
                     w2=beta2_cv, w3=beta3_cv, w4=beta4_cv)
  
  nonzero <- list(lasso = which(beta_lasso_cv != 0),
                  w1 = which(beta1_cv > 10^{-4}),
                  w2 = which(beta2_cv > 10^{-4}),
                  w3 = which(beta3_cv > 10^{-5}),
                  w4 = which(beta4_cv > 10^{-4}))
  lapply(nonzero, length)
  
  #nonzero1 <- list(lasso = which(beta_lasso_cv != 0),
  #                 w1 = which(beta1_cv != 0),
  #                 w2 = which(beta2_cv != 0),
  #                 w3 = which(beta2_cv != 0),
  #                 w4 = which(beta4_cv != 0))
  #lapply(nonzero1, length)
  
  zero <- list(w1 = which(beta$w1 <= 10^{-4}),
               w2 = which(beta$w2 <= 10^{-4}),
               w3 = which(beta$w3 <= 10^{-5}),
               w4 = which(beta$w4 <= 10^{-4}))
  
  beta$w1[zero$w1]=0
  beta$w2[zero$w2]=0
  beta$w3[zero$w3]=0
  beta$w4[zero$w4]=0
  
  ##error rate
  error_lasso <- cal_mis(X, beta_lasso_cv, y)
  error1 <- cal_mis(X, beta1_cv, y)
  error2 <- cal_mis(X, beta2_cv, y)
  error3 <- cal_mis(X, beta3_cv, y)
  error4 <- cal_mis(X, beta4_cv, y)
  
  error_onetime <- c(error_lasso,error1,error2,error3,error4)
  error_onetime
  
  ##show gene symbol
  gene <- colnames(X)
  gene_out <- list(lasso=gene[nonzero$lasso],
                   w1=gene[nonzero$w1],
                   w2=gene[nonzero$w2],
                   w3=gene[nonzero$w3],
                   w4=gene[nonzero$w4])
  write.csv(gene_out$lasso, file="~/Desktop/paper_Zhang/outcome/newoutcome/2_gene_out_lasso.csv")
  write.csv(gene_out$w1, file="~/Desktop/paper_Zhang/outcome/newoutcome/2_gene_out_w1.csv")
  write.csv(gene_out$w2, file="~/Desktop/paper_Zhang/outcome/newoutcome/2_gene_out_w2.csv")
  write.csv(gene_out$w3, file="~/Desktop/paper_Zhang/outcome/newoutcome/2_gene_out_w3.csv")
  write.csv(gene_out$w4, file="~/Desktop/paper_Zhang/outcome/newoutcome/2_gene_out_w4.csv")
  

  
  ##2. leave-one-out error rate
  modelsize <- matrix(0, nrow=nrow(X),ncol=5)
  error_matrix <- matrix(0, nrow=nrow(X), ncol=5)
  for(k in 1:nrow(X)){
    X_loo <- X[-k,]  
    y_loo <- y[-k]
    
    p <- ncol(X_loo)
    n_train <- nrow(X_loo)
    X_std <- scale(X_loo) / sqrt(n_train - 1) * sqrt(n_train)
    r=1
    
    ##use cv.glmnet to select best lambda
    m3 <- cv.glmnet(X_std, y_loo, family = "binomial", alpha = 1)
    #nfold = 10 (default)
    lam_cv <- m3$lambda.min
    cat('iteration', k, '\n')
    cat('optimal lambda by cv of glmnet: ', lam_cv, '\n')
    
    lam <- lam_cv
    
    ## use glmnet to find beta_lasso under the lam_cv. 
    m2_cv <- glmnet(X_std, y_loo, lambda = lam, family = "binomial", alpha = 1)
    beta_lasso_cv <- m2_cv$beta[,1]
    
    ## weight 1
    w1 <- cal_weight_1(X_loo, y_loo, r)
    beta1_cv <- WLOG(X_loo, y_loo, w1, lam)$beta
    
    ## weight 2
    w2 <- cal_weight_2(X_loo, y_loo, r)
    beta2_cv <- WLOG(X_loo, y_loo, w2, lam)$beta 
    
    ## weight 3
    w3 <- cal_weight_3(X_loo, y_loo, r)
    beta3_cv <- w3_WLOG(X_loo, y_loo, w3, lam)
    
    ##weight 4
    w4 <- cal_weight_4(X_loo, y_loo, lam)
    beta4_cv <- WLOG(X_loo, y_loo, w4, lam)$beta
    
    zero <- list(lasso = which(beta_lasso_cv == 0),
                 w1 = which(beta1_cv <= 1*10^{-4}),
                 w2 = which(beta2_cv <= 1*10^{-4}),
                 w3 = which(beta3_cv <= 1*10^{-4}),
                 w4 = which(beta4_cv <= 1*10^{-4}))
    ##modelsize
    modelsize[k,1] <- ncol(X)-length(zero$lasso)
    modelsize[k,2] <- ncol(X)-length(zero$w1)
    modelsize[k,3] <- ncol(X)-length(zero$w2)
    modelsize[k,4] <- ncol(X)-length(zero$w3)
    modelsize[k,5] <- ncol(X)-length(zero$w4)
    
    beta_lasso_cv[zero$lasso]=0
    beta1_cv[zero$w1]=0
    beta2_cv[zero$w2]=0
    beta3_cv[zero$w3]=0
    beta4_cv[zero$w4]=0
    
    ##error rate
    error_matrix[k,1]  <- cal_mis(X[k,], beta_lasso_cv, y[k])
    error_matrix[k,2] <- cal_mis(X[k,], beta1_cv, y[k])
    error_matrix[k,3] <- cal_mis(X[k,], beta2_cv, y[k])
    error_matrix[k,4] <- cal_mis(X[k,], beta3_cv, y[k])
    error_matrix[k,5] <- cal_mis(X[k,], beta4_cv, y[k])
  }
  
  mean_ms <- apply(modelsize,2,mean)
  mean_error <- apply(error_matrix,2,mean)
  sd_ms <- apply(modelsize,2,sd)
  sd_error <- apply(error_matrix,2,sd)
  
  names(mean_ms) <- c("lasso","w1","w2","w3","w4")
  names(mean_error) <- c("lasso","w1","w2","w3","w4")
  names(sd_ms) <- c("lasso","w1","w2","w3","w4")
  names(sd_error) <- c("lasso","w1","w2","w3","w4")
  
  mean_ms
  mean_error
  sd_ms
  sd_error 

  }


learnModel <- function(data, labels){
  
  lambda = 0.1
  alpha = 0.05
  n = ncol(data)
  m = nrow(data)
  all_theta = matrix(0,0,n+1)
  
  X = cbind(matrix(1,m,1), data)
  for (digit in 0:9) {
    init_theta = matrix(0, 1, n + 1)
    theta = gradientDescent(X, ifelse(labels == digit, 1, 0), init_theta, lambda, alpha, 100)
    all_theta = rbind(all_theta, theta)
  }
  
  return(all_theta)
}

testModel <- function(classifier, trainData){
  n = ncol(trainData)
  m = nrow(trainData)
  X = cbind(matrix(1,m,1), trainData)
  
  labels = matrix(apply(sigmoid(X %*% t(classifier)), 1, which.max)-1, m, 1)
  
  return(labels)
}

# ALready concatenated first feature with 1`s
# theta - 1 * (n+1)
# X - m * (n + 1)
costFunction <- function(theta, X, y, lambda){
  m = nrow(y)
  
  sigmoidTemp <- clip(sigmoid(X%*%t(theta)))
  
  thetaTemp <- theta;
  thetaTemp[1] = 0;
  reg = lambda/(2*m)*rowSums(thetaTemp^2);
  
  terms = y * log(sigmoidTemp)+(1-y)*log(1-sigmoidTemp);
  
  J <- -1/m*(colSums(terms)) + reg;
  
  return(J);
}

gradient <- function(theta, X, y, lambda){
  sigmoidTemp <- clip(sigmoid(X%*%t(theta)))
  thetaTemp <- theta;
  thetaTemp[1] = 0;
  m = nrow(y)
  n = ncol(X)
  
  grad = 1/m * (colSums(matrix((sigmoidTemp - y),m,n)*X)+ lambda * thetaTemp);
  return(grad)
}

gradientDescent <- function(X, y, theta, lambda, alpha, num_iters){
  
  for (i in 1:num_iters) {
    theta = theta - alpha * gradient(theta, X, y, lambda);
  }
  
  return(theta)
}

clip <- function(x){
  result <- ifelse(x==0, 1e-15, x)
  result <- ifelse(result==1, 1-1e-15, result)
  return(result)
}

sigmoid <- function(x){
  g <- 1.0 / (1.0 + exp(-x));
  return(g);
}
learnModel <- function(data, labels){
  
  Rprof(tmp <- tempfile(),interval = 0.001)
  
  lambda = 0.01
  alpha = 0.001
  epsilon = 0.1
  delta = 0.95
  num_iters = 100
  n = ncol(data)
  m = nrow(data)
  all_theta = matrix(0,0,n+1)
  
  X = cbind(matrix(1,m,1), data)
  init_theta = matrix(0, 1, n + 1)
  labels = labels
  
  for (digit in 9:9) {
    print(paste0('Digit: ', digit))
    GD = gradientDescent(X, labels == digit, init_theta, lambda, alpha, epsilon, delta, num_iters)
    theta = GD$theta;
    plot(1:num_iters, GD$costFromIter)
    all_theta = rbind(all_theta, theta)
  }
  
  
  Rprof()
  MyTimerTranspose=summaryRprof(tmp)$sampling.time
  unlink(tmp)
  
  print(paste0('Time: ', MyTimerTranspose))
  
  return(all_theta)
}

testModel <- function(classifier, trainData){
  n = ncol(trainData)
  m = nrow(trainData)
  X = cbind(matrix(1,m,1), trainData)
  
  labels = sigmoid(X %*% t(classifier)) > 0.5
  return(labels)
}

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
  
  grad = 1/m * (colSums(as.vector(sigmoidTemp - y)*X)+ lambda * thetaTemp);
  return(grad)
}

gradientDescent <- function(X, y, theta, lambda, alpha, epsilon, delta, num_iters){
  costs <- double(num_iters)
  prev = costFunction(theta, X, y, lambda)
  
  
  for (i in 1:num_iters) {
    grad = gradient(theta, X, y, lambda)
    theta = theta - alpha * grad
  
    costs[i] = costFunction(theta, X, y, lambda)
    
    if(costs[i] > prev-epsilon*alpha*sum(grad^2))
      alpha = delta*alpha
    
    if (abs(prev - costs[i]) < 1e-7)
      break
    
    prev = costs[i]
  }
  
  return(list("theta" = theta, "costFromIter" = costs))
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
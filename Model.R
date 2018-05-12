learnModel <- function(data, labels){
  
  
  no_cores <- detectCores() - 5
 
  cl <- makeCluster(no_cores)
  clusterExport(cl, "gradientDescent")
  clusterExport(cl, "gradient")
  clusterExport(cl, "costFunction")
  clusterExport(cl, "sigmoid")
  clusterExport(cl, "clip")
  
  lambda = 0.1
  alpha = 0.005
  epsilon = 0.1
  delta = 0.95
  num_iters = 150
  
  n = ncol(data)
  m = nrow(data)
  
  X = cbind(matrix(1,m,1), data)
  init_theta = matrix(0, 1, n + 1)
  labels = labels

  all_theta = parSapply(cl, 0:9,
            function(digit){
              gradientDescent(X, labels == digit, init_theta, lambda, alpha, epsilon, delta, num_iters)
            })
           
  stopCluster(cl)
  return(t(all_theta))
}

testModel <- function(classifier, trainData){
  n = ncol(trainData)
  m = nrow(trainData)
  X = cbind(matrix(1,m,1), trainData)
  
  labels = matrix(apply(sigmoid(X %*% t(classifier)), 1, which.max)-1, m, 1)
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
  prev = costFunction(theta, X, y, lambda)
  
  for (i in 1:num_iters) {
    grad = gradient(theta, X, y, lambda)
    theta = theta - alpha * grad
    
    cost = costFunction(theta, X, y, lambda)
    
    if(cost > prev-epsilon*alpha*sum(grad^2))
      alpha = delta*alpha
    
    if (abs(prev - cost) < 1e-7)
      break
    
    prev = cost
  }
  return( theta)
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
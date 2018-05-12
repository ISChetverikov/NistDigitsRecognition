learnModel <- function(data, labels){
  # Fits the model parameters
  #
  # Args:
  # data: m*n matrix of train data
  # labels: m*1 matrix of train label 
  
  # Returns:
  # (number of labels)*(n+1) matrix of model parameters
  no_cores = 3 # cores count for cluster
 
  # set up cluster
  cl <- makeCluster(no_cores)
  clusterExport(cl, "gradientDescent")
  clusterExport(cl, "gradient")
  clusterExport(cl, "costFunction")
  clusterExport(cl, "sigmoid")
  clusterExport(cl, "clip")
  
  # gradient descent parameters
  lambda = 0.1
  alpha = 0.005
  epsilon = 0.1
  delta = 0.95
  num_iters = 20
  
  digits = 0:9
  
  n = ncol(data) # features count
  m = nrow(data) # examples count
  
  X = cbind(matrix(1,m,1), data) # expand data with first column of ones
  init_theta = matrix(0, 1, n + 1) # zero-filled initial theta vector
  labels = labels # locate labels as local variable (for cluster)

  # Parallel applying of GD with one vs all training scheme 
  all_theta = parSapply(cl, digits,
            function(digit){
              gradientDescent(X, labels == digit, init_theta, lambda, alpha, epsilon, delta, num_iters)
            })
           
  stopCluster(cl)
  
  return(t(all_theta))
}

testModel <- function(all_theta, data){
  # Predicts labels on data
  #
  # Args:
  # all_theta: (number of labels)*(n+1) matrix of model parameters
  # data: m*n matrix of data
  
  # Returns:
  # m*1 matrix of predicted labels
  n = ncol(data)
  m = nrow(data)
  X = cbind(matrix(1,m,1), data)
  
  labels = matrix(apply(sigmoid(X %*% t(all_theta)), 1, which.max)-1, m, 1)
  
  return(labels)
}

gradientDescent <- function(X, y, theta, lambda, alpha, epsilon, delta, num_iters){
  # Gradient descent algorithm with adaptive learning rate
  #
  # Args:
  # theta: 1*(n+1) matrix of model parameters
  # X: m*(n+1) matrix of train data
  # y: m*1 matrix of train binary labels
  # lambda: regularization coefficient
  # alpha: start value of learning rate
  # epsilon: constant for condition of learning rate decreasing
  # delta: coefficient of learning rate decreasing
  # num_iters: max number of iterations
  
  # Returns:
  # 1*(n+1) matrix of optimal theta that minimizes cost function
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
  return(theta)
}


gradient <- function(theta, X, y, lambda){
  # Computes gradient of cost function
  #
  # Args:
  # theta: 1*(n+1) matrix of model parameters
  # X: m*(n+1) matrix of train data
  # y: m*1 matrix of train labels
  # lambda: regularization coefficient
  
  # Returns:
  # 1*(n+1) matrix of cost function gradient
  sigmoidTemp <- clip(sigmoid(X%*%t(theta)))
  thetaTemp <- theta;
  thetaTemp[1] = 0;
  
  m = nrow(y)
  
  grad = 1/m * (colSums(as.vector(sigmoidTemp - y) * X) + lambda * thetaTemp);
  
  return(grad)
}

costFunction <- function(theta, X, y, lambda){
  # Computes cost function of logistic regression
  #
  # Args:
  # theta: 1*(n+1) matrix of model parameters
  # X: m*(n+1) matrix of train data
  # y: m*1 matrix of train labels
  
  # Returns:
  # Value of logistic regression cost function 
  m = nrow(y)
  
  sigmoidTemp <- clip(sigmoid(X%*%t(theta)))
  
  thetaTemp = theta
  thetaTemp[1] = 0
  reg = lambda/(2*m)*rowSums(thetaTemp^2)
  
  terms = y * log(sigmoidTemp)+(1-y)*log(1-sigmoidTemp)
  
  J = -1/m*(colSums(terms)) + reg;
  
  return(J);
}

clip <- function(x){
  # Clips value between 0+EPSILON and 1-EPSILON
  # (It needs for correct computing of logloss function)
  #
  # Args:
  # x: real value
  
  # Returns:
  # If x arg is less or equal 0 then result is 0+EPSILON
  # if x arg is greater or equal 1 then result is 1-EPSILON
  # else result is equal x arg
  EPSILON = 1e-15
  
  result = ifelse(x<=0, 0+EPSILON , x)
  result = ifelse(result>=1, 1-EPSILON, result)
  
  return(result)
}


sigmoid <- function(x){
  # Computes logistic function 
  #
  # Args:
  # x: real value
  
  # Returns:
  # Result of logistic function computing with input arg x
  g = 1.0 / (1.0 + exp(-x));
  
  return(g);
}
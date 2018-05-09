learnModel <- function(data, labels){
  
  return(0);
}

testModel <- function(classifier, trainData){
  
  labels <- matrix(data = 1, nrow = nrow(trainData), ncol = 1)
  
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

clip <- function(x){
  result <- ifelse(x==0, 1e-15, x)
  result <- ifelse(result==1, 1-1e-15, result)
  return(result)
}

sigmoid <- function(x){
  g <- 1.0 / (1.0 + exp(-x));
  return(g);
}
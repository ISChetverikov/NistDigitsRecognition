learnModel <- function(data, labels){
  # Нахождение параметров модели для 1 цифры
  #
  # Args:
  # data: m*n матрица признаков обучающих примеров
  # labels: m*1 матрица значений целевой переменной бинарной классификации обучающих примеров
  
  # Returns:
  # (1)*(n+1) вектор параметров модели
  digit = 7
  
  lambda = 0.01
  alpha = 0.001
  epsilon = 0.1
  delta = 0.95
  num_iters = 100
  n = ncol(data)
  m = nrow(data)
  
  X = cbind(matrix(1,m,1), data)
  init_theta = matrix(0, 1, n + 1)
  labels = labels
  
  print(paste0('Digit: ', digit))
  GD = gradientDescent(X, labels == digit, init_theta, lambda, alpha, epsilon, delta, num_iters)
  theta = GD$theta
  plot(1:num_iters, GD$costFromIter)
  
  return(theta)
}

testModel <- function(theta, trainData){
  # Предсказание значения целевой переменной бинарной классификации на данных
  #
  # Args:
  # all_theta: (1)*(n+1) вектор параметров модели
  # data: m*n матрица признаков примеров
  
  # Returns:
  # m*1 вектор предсказанных значений
  n = ncol(trainData)
  m = nrow(trainData)
  X = cbind(matrix(1,m,1), trainData)
  
  labels = sigmoid(X %*% t(theta)) > 0.5
  return(labels)
}

gradientDescent <- function(X, y, theta, lambda, alpha, epsilon, delta, num_iters){
  # Метод градиентного спуска с дробящимся шагом
  #
  # Args:
  # theta: 1*(n+1) вектор параметров модели
  # X: m*(n+1) матрица обучающих примеров
  # y: m*1 вектор значений целевой переменной (принадлежность одному из двух классов)
  # lambda: коэффициент регуляризации
  # alpha: начальное значение темпа обучения
  # epsilon: константа для условия уменьшения темпа обучения
  # delta: коэффициент уменьшения темпа обучения
  # num_iters: максимальное число операций
  
  # Returns:
  #     List:
  #     "theta": 1*(n+1) вектор оптимальных значений, минимизирующих функцию потерь
  #     "costFromIter": массив значений функции потерь для каждой итерации
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

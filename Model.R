learnModel <- function(data, labels){
  # Нахождение параметров модели для 10 цифр
  #
  # Args:
  # data: m*n матрица признаков обучающих примеров
  # labels: m*1 матрица значений целевой переменной обучающий примеров
  
  # Returns:
  # (10)*(n+1) матрица параметров модели
  library(parallel)
  
  no_cores = 3 # количество ядер для параллельных вычислений
 
  # настройка параллельных вычислений
  cl <- makeCluster(no_cores)
  clusterExport(cl, "gradientDescent")
  clusterExport(cl, "gradient")
  clusterExport(cl, "costFunction")
  clusterExport(cl, "sigmoid")
  clusterExport(cl, "clip")
  
  # параметры градиентного спуска
  lambda = 0.1
  alpha = 0.005
  epsilon = 0.1
  delta = 0.95
  num_iters = 1
  
  # массив меток-цифр
  digits = 0:9
  
  n = ncol(data) # количество признаков
  m = nrow(data) # количество примеров
  
  X = cbind(matrix(1,m,1), data) # расширение матрицы признаков первым столбцом с единицами
  init_theta = matrix(0, 1, n + 1) # начальный вектор параметров заполнен нулями
  labels = labels # столбец значений целевой переменной как локальная переменная
                  # (нужно для параллельных вычислений)

  # Параллельное вычисление параметров модели для каждой метки-цифры
  all_theta = parSapply(cl, digits,
            function(digit){
              gradientDescent(X, labels == digit, init_theta, lambda, alpha, epsilon, delta, num_iters)
            })
           
  stopCluster(cl)
  
  return(t(all_theta))
}

testModel <- function(all_theta, data){
  # Предсказание значения целевой переменной на данных
  #
  # Args:
  # all_theta: (10)*(n+1) матрица параметров модели
  # data: m*n матрица признаков примеров
  
  # Returns:
  # m*1 матрица предсказанных значений
  n = ncol(data)
  m = nrow(data)
  X = cbind(matrix(1,m,1), data)
  
  labels = matrix(apply(sigmoid(X %*% t(all_theta)), 1, which.max)-1, m, 1)
  
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
  # 1*(n+1) вектор оптимальных значений, минимизирующих функцию потерь
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



gradient <- function(theta, X, y, lambda){
  # Вычисление градиента функции потерь
  #
  # Args:
  # theta: 1*(n+1) вектор параметров модели
  # X: m*(n+1) матрица признаков обучающих примеров
  # y: m*1 вектор значений целевой переменной
  # lambda: коэффициент регуляризации
  
  # Returns:
  # 1*(n+1) вектор градиента функции потерь
  sigmoidTemp <- clip(sigmoid(X%*%t(theta)))
  thetaTemp <- theta;
  thetaTemp[1] = 0;
  
  m = nrow(y)
  
  grad = 1/m * (colSums(as.vector(sigmoidTemp - y) * X) + lambda * thetaTemp);
  
  return(grad)
}

costFunction <- function(theta, X, y, lambda){
  # Вычисление функции потерь логистической регрессии
  #
  # Args:
  # theta: 1*(n+1) вектор параметров модели
  # X: m*(n+1) матрица признаков обучающих примеров
  # y: m*1 значений целевой переменной обучающих примеров
  
  # Returns:
  # Значение функции потерь логистической регрессии
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
  # "Зажимает" значение аргумента между 0+EPSILON и 1-EPSILON
  # (Нужно для коректного взятия логарифма от нулевых значений)
  #
  # Args:
  # x: аргумент
  
  # Returns:
  # Если x меньше или равен 0: 0+EPSILON
  # Если x больше или равен 1: 1-EPSILON
  # Иначе: x
  EPSILON = 1e-15
  
  result = ifelse(x<=0, 0+EPSILON , x)
  result = ifelse(result>=1, 1-EPSILON, result)
  
  return(result)
}


sigmoid <- function(x){
  # Вычислене значения логистической функции 
  #
  # Args:
  # x: аргумент функции
  
  # Returns:
  # Значение логистической функции при аргумент x
  g = 1.0 / (1.0 + exp(-x));
  
  return(g);
}
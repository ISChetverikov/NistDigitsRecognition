source("load_data.R")

source("SharedFunctions.R")

# Запуск решения для многоклассовой классификации 10 цифр
source("Model.R")
source("ClassificationMain.R")

# Запуск решения бинарной классификации для 1 цифры
#source("Model_Binary.R")
#source("ClassificationMain_Binary.R")

# Чистка памяти
gc()
rm(list = ls(all.names = TRUE))
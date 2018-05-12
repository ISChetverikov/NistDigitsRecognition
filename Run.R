source("load_data.R")

source("Model.R")
#source("Model_DEBUG.R")
source("ClassificationMain.R")
#source("ClassificationMain_DEBUG.R")

# memory cleaning
gc()
rm(list = ls(all.names = TRUE))
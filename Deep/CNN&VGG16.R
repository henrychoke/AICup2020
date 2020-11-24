## This file is tested under R 3.6.3 in Win 10 and RStudio Version 1.2.5042
## The installation should be implimented in the campus of NCKU 
## to avoid some errors from web connection.


##########################################################################
# Chapter 3
# install.packages("keras")
# install.packages("tensorflow")
# install.packages("magrittr")
# install.packages("ggplot2")
# install.packages("reticulate")
# install.packages("dplyr")

library(reticulate) 
library(keras)
library(tensorflow)
# install_miniconda()
# install_tensorflow()
# do not use install_keras()


library(dplyr)
library(ggplot2)

library(magrittr)
library(pbapply)


#==========================================================================
# Chapter 7 Convolutional Neural Network 
#=========================================================================
#=========================================================================
# 7.3 Training a ConvNet on a Small DataSet Using keras

# Need to restart R

#========================================================
#¤À³Î¸ê®Æ
library(EBImage)
library(imager)
df <- read.csv("C:/Users/user/Desktop/Aldea/secondrelease/train.csv",                 
               header = T,colClasses = "character")
df <- read.csv("C:/Users/user/Desktop/Aldea/secondrelease/dev.csv",                 
               header = T,colClasses = "character")
path.cut <- "C:/Users/user/Desktop/Aldea/secondrelease/opencvoutputdev/" 
train.labelA <- "C:/Users/user/Desktop/Aldea/deep/full/train/A/"
train.labelB <- "C:/Users/user/Desktop/Aldea/deep/full/train/B/"
train.labelC <- "C:/Users/user/Desktop/Aldea/deep/full/train/C/"
test.labelA <- "C:/Users/user/Desktop/Aldea/deep/full/test/A/"
test.labelB <- "C:/Users/user/Desktop/Aldea/deep/full/test/B/"
test.labelC <- "C:/Users/user/Desktop/Aldea/deep/full/test/C/"
attach(df)
set.seed(20)
dfsubsetA <- df[label == "A" ,];dfsubsetB <- df[label == "B" ,];dfsubsetC <- df[label == "C" ,]
a=dfsubsetA[,-2];b=dfsubsetB[,-2];c=dfsubsetC[,-2]
sample.a = sample(a,500);sample.b = sample(b,500);sample.c = sample(c,500)
for (i in 1:dim(dfsubsetA)[1]){
  file <- dfsubsetA[i,1]
  xaa <- load.image(file = paste(path.cut,file,sep="")) 
  outputpath=paste(test.labelA,file,sep="")
  save.image(xaa,outputpath)
}
for (i in 1:dim(dfsubsetB)[1]){
  file <- dfsubsetB[i,1]
  xaa <- load.image(file = paste(path.cut,file,sep="")) 
  outputpath=paste(test.labelB,file,sep="")
  save.image(xaa,outputpath)
}  
for (i in 1:dim(dfsubsetC)[1]){
  file <- dfsubsetC[i,1]
  xaa <- load.image(file = paste(path.cut,file,sep="")) 
  outputpath=paste(test.labelC,file,sep="")
  save.image(xaa,outputpath)
}
#================================================================================

new_dir <- "C:/Users/user/Desktop/Aldea/deep/full"

train_dir_path <- file.path(new_dir, "train")
validation_dir_path <- file.path(new_dir, "validation")
test_dir_path <- file.path(new_dir, "test")
train_A_dir <- file.path(train_dir_path, "A")
train_B_dir <- file.path(train_dir_path, "B")
train_C_dir <- file.path(train_dir_path, "C")
validation_A_dir <- file.path(validation_dir_path, "A")
validation_B_dir <- file.path(validation_dir_path, "B")
validation_C_dir <- file.path(validation_dir_path, "C")
test_A_dir <- file.path(test_dir_path, "A")
test_B_dir <- file.path(test_dir_path, "B")
test_C_dir <- file.path(test_dir_path, "C")


library(keras)
is_keras_available()
img_width <- 230
img_height <- 230
batch_size <- 32
epochs <- 30
train_samples = 5610
validation_samples = 800

train_generator <- flow_images_from_directory(
  train_dir_path, generator = image_data_generator(rescale = 1 / 255),
  target_size = c(img_height, img_width), color_mode = "rgb",
  class_mode = "categorical", batch_size = batch_size,
  shuffle = TRUE, seed = 61)

validation_generator <- flow_images_from_directory(
  validation_dir_path, generator = image_data_generator(rescale = 1 / 255),
  target_size = c(img_width, img_height), color_mode = "rgb",
  classes = NULL, class_mode = "categorical", batch_size = batch_size,
  shuffle = TRUE, seed = 61)

## a CONV layer having 32 filters of size 3 ?? 3, followed
## by relu activation and a POOL layer with max_pooling
model_7 <- keras_model_sequential()
model_7 %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3),
                input_shape = c(img_height, img_width, 3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 64, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 128, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 128, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")
summary(model_7)

#conda_install(envname="C:\\Users\\user\\Anaconda3\\envs\\r-reticulate",
#             packages=c("pillow", "scipy"))    

#conda_install(envname="C:\\Users\\stat\\AppData\\Local\\r-miniconda\\envs\\r-reticulate",
#             packages=c("pillow"))   

## compile and fit the model.
model_7 %>% compile(loss = "categorical_crossentropy", optimizer = 
                      optimizer_adam(lr = 0.001, decay = 1e-06), metrics = "accuracy")

history_7 <- model_7 %>% fit_generator(train_generator, steps_per_epoch = as.integer(train_samples/batch_size),
                                       epochs = epochs, validation_data = validation_generator,
                                       validation_steps = as.integer(validation_samples/batch_size),
                                       verbose = 2)

plot(history_7)

evaluate_generator(model_7, validation_generator, validation_samples)
# loss accuracy 
# 2.619864 0.694000

getwd()
model_path="C:/Users/user/Desktop/Aldea/deep/subset"
save_model_hdf5(model_7, paste0(model_path, "basic_cnn.h5"),
                overwrite = TRUE)
save_model_weights_hdf5(model_7, paste0(model_path, "basic_cnn_weights.h5"),
                        overwrite = TRUE)

#=============================================================================
## 7.3.1 Data Augmentation

augment <- image_data_generator(rescale = 1/255, rotation_range = 50,
                                width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2,
                                zoom_range = 0.2, horizontal_flip = TRUE, fill_mode = "nearest")
train_generator_augmented <- flow_images_from_directory(
  train_dir_path, generator = augment,
  target_size = c(img_height, img_width),
  color_mode = "rgb", class_mode = "categorical", batch_size = batch_size,
  shuffle = TRUE, seed = 61)

validation_generator <- flow_images_from_directory(
  validation_dir_path, generator = image_data_generator(rescale = 1 / 255),
  target_size = c(img_height, img_width), color_mode = "rgb",
  classes = NULL, class_mode = "categorical", batch_size = batch_size,
  shuffle = TRUE, seed = 61)

batch <- generator_next(train_generator_augmented)
str(batch)

sp <- par(mfrow = c(3, 3), pty = "s", mar = c(1, 0, 1, 0))
for (i in 1:9) {
  batch <- generator_next(train_generator_augmented)
  plot(as.raster(batch[[1]][1, , , ]))
}

par(sp)

model_7_1 <- keras_model_sequential()
model_7_1 %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3),
                input_shape = c(img_height, img_width, 3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 64, kernel_size = c(3,3)) %>% #32
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 128, kernel_size = c(3,3)) %>% #64
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filter = 128, kernel_size = c(3,3)) %>% #64
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")

model_7_1 %>% compile(loss = "categorical_crossentropy", optimizer = optimizer_adam(lr = 1e-04,
                                                                               decay = 1e-06), metrics = "accuracy")
history_7_1 <- model_7_1 %>% fit_generator(train_generator_augmented,
                                           steps_per_epoch = as.integer(train_samples/batch_size), epochs = epochs,
                                           validation_data = validation_generator, validation_steps = as.integer(validation_samples/batch_size),
                                           verbose = 2)

plot(history_7_1)
evaluate_generator(model_7_1, validation_generator, validation_samples)

save_model_weights_hdf5(model_7_1, paste0(model_path, "augmented_cnn_weights.h5"),
                        overwrite = TRUE)
save_model_hdf5(model_7_1, paste0(model_path, "augmented_cnn.h5"),
                overwrite = TRUE)

#=========================================================================
# 7.4 Specialized Neural Network Architectures


#===================================================
# 7.4.5 Transfer Learning or Using Pretrained Models

# In keras, we have a handful of pretrained models in the ImageNet dataset (containing
# 1.2 million images with 1000 categories), and they are?XVGG16, VGG19,
# MobileNet, ResNet50, etc.

# conv_vgg <- application_vgg16(
#   weights = "imagenet",
#   include_top = F,
#   input_shape = c(150, 150, 3)
# )
# conv_vgg
# 
# #===================================================
# # 7.4.6 Feature Extraction
# 
# # base_dir <- "H:/Data/Kaggle/PetImagesSubset"
# train_dir <- file.path(base_dir, "train")
# validation_dir <- file.path(base_dir, "validation")
# test_dir <- file.path(base_dir, "test")
# data_gen <- image_data_generator(rescale = 1 / 255)
# batch_size <- 20
# 
# extracted_features <- function(directory, sample){
#   features <- array(0, dim = c(sample, 4, 4, 512))
#   labels <- array(0, dim = c(sample))
#   generator <- flow_images_from_directory(
#     directory = directory,
#     generator = data_gen,
#     target_size = c(150, 150),
#     batch_size = batch_size,
#     class_mode = "binary"
#   )
#   i <- 0
#   while(TRUE) {
#     batch <- generator_next(generator)
#     input_batch <- batch[[1]]
#     label_batch <- batch[[2]]
#     feature_batch <- conv_vgg %>% predict(input_batch)
#     index_range <- ((i * batch_size) + 1):((i + 1) * batch_size)
#     features[index_range,,,] <- feature_batch
#     labels[index_range] <- label_batch
#     i <- i + 1
#     if(i * batch_size >= sample) break
#   }
#   list(
#     features = features,
#     labels = labels
#   )
# }
# 
# train <- extracted_features(train_dir, 1000)
# validation <- extracted_features(validation_dir, 500)
# test <- extracted_features(test_dir, 500)
# 
# reshape_features <- function(features){
#   array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
# }
# 
# train$features <- reshape_features(train$features)
# validation$features <- reshape_features(validation$features)
# test$features <- reshape_features(test$features)
# 
# model_7_2 <- keras_model_sequential() %>%
#   layer_dense(units = 256, activation = "relu", input_shape = 4 * 4 * 512) %>%
#   layer_dropout(rate = 0.5) %>%
#   layer_dense(units = 1, activation = "sigmoid")
# 
# model_7_2 %>% compile(
#   optimizer = optimizer_rmsprop(lr = 2e-5),
#   loss = "binary_crossentropy",
#   metrics = "accuracy"
# )
# 
# history_7_2 <- model_7_2 %>% fit(
#   x = train$features, y = train$labels,
#   epochs = 30,
#   batch_size = 20,
#   validation_data = list(validation$features, validation$labels)
# )
# 
# plot(history_7_2)
# 
# evaluate(model_7_2, test$features, test$labels)
# 
# save_model_weights_hdf5(model_7_2,
#                         paste0(model_path,
#                                "featureExtract_vgg16_weights.h5"),
#                         overwrite = TRUE)
# save_model_hdf5(model_7_2, paste0(model_path, "featureExtract_vgg16.h5"),
#                 overwrite = TRUE)
# 
# #model_7_2 <- load_model_hdf5(paste0(model_path, "featureExtract_vgg16.h5"))
# 
# save.image(file="E:R_deeplearning_CH7_20200531.Rdata")
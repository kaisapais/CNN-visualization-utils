# CNN-visualization-utils

Utility functions for visualizing properties of CNN model (made with python's keras module). For now, only hidden layer output visualization. Function saves outputs as png files to given folder.

Usage: cnn_outputs.visualizeOutputs(model,image)

image shape should match model input (todo: resize)

image.shape = (y,x,n) or (y,x)

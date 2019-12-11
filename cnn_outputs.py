# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:23:40 2019

@author: kaisapais
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import keras.backend as K

import os

def checkDir(dirname):
    #Create directory if does not exist
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        print('created directory',dirname)


def getOutputs(model,x_test):
    # Get output images of all layers
    outputs = [np.zeros((1,))]
    numLayers = len(model.layers)
    
    for i in range(1,numLayers):
        if model.layers[i].name[0:7] == 'dropout':
            outputs.append(np.zeros((1,)))
        else:
            get_output = K.function([model.layers[0].input,K.learning_phase()],[model.layers[i].output])
            thisRes = get_output([x_test,0])[0]
            #print(model.layers[i].name,thisRes.shape)
            outputs.append(thisRes)
        
    return outputs

def normalize(im,imin=0,imax=1):
    im = im.astype('float')
    return (im-np.min(im))*((imax-imin)/(np.max(im)-np.min(im))) + imin



def visualizeOutputs(model,im,colormap='cubehelix',outputfolder='outputs_tmp/'):
    im = normalize(im)
    if outputfolder[-1] == '/':
        pass
    else:
        outputfolder = outputfolder+'/'
    
    checkDir(outputfolder)
    
    cmap = plt.get_cmap(colormap)
    
    if len(im.shape) == 2:
        x_test = np.expand_dims(im,axis=0)
        x_test = np.expand_dims(x_test,axis=-1)
    else:        
        x_test = np.expand_dims(im,axis=0)
    
    if len(im.shape) == 3 and im.shape[-1] == 3:#draw as rgb
        im = (im*255).astype('uint8')
        img = Image.fromarray(im)
        img.save(outputfolder+'input.png')
    else:
        for i in range(x_test.shape[-1]):
            patch = cmap(x_test[0,...,i])
            patch = np.delete(patch, 3, 2)
            img = Image.fromarray((patch*255).astype('uint8'))
            img.save(outputfolder+'input_'+str(i)+'.png')

    outputs = getOutputs(model,x_test)
    layers = model.layers
    for i,l in enumerate(layers):
        op = outputs[i]
        ln = l.name
        if ln[0:4] == 'conv':
            for j in range(op.shape[-1]):
                
                patch = op[0,...,j]
                patch = normalize(patch)
                
                patch = cmap(patch)
                patch = np.delete(patch, 3, 2)
                img = Image.fromarray((patch*255).astype('uint8'))
                img.save(outputfolder+ln+'_'+str(j)+'.png')

        
    
        



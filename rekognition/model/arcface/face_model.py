# Part of the code used is given under MIT License
# Copyright (c) 2018 Jiankang Deng and Jia Guo

import os
import numpy as np
import mxnet as mx
import cv2
import sklearn
from .face_preprocess import preprocess

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, model):
    ctx = mx.gpu(0)
    _vec = "112,112".split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    self.model = get_model(ctx, image_size, model, 'fc1')

    self.threshold = "1.24"
    self.det_minsize = 50
    self.det_threshold = [0.6,0.7,0.8]
    #self.det_factor = 0.9
    self.image_size = image_size

  def get_input(self, face_img):
    nimg = preprocess(face_img, image_size='112,112')
    if len(nimg.shape) == 4:  # Batch preprocessing
      ret_batch = []
      for i in range(len(nimg)):
        _img = cv2.cvtColor(nimg[i], cv2.COLOR_BGR2RGB)
        batch_img = np.transpose(_img, (2,0,1))

        ret_batch.append(batch_img)
      aligned = np.array(ret_batch)
    else:
      nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
    return aligned

  def get_feature(self, aligned):
    if len(aligned.shape) == 4:
      embedding = self.inference(aligned)
    else:
      input_blob = np.expand_dims(aligned, axis=0)
      embedding = self.inference(input_blob)

    if len(aligned.shape) != 4:
      embedding = embedding.flatten()

    return embedding

  def inference(self, input_blob):
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding)

    return embedding
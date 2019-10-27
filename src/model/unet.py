# -*- coding: utf-8 -*-
import keras
from model.common import *
from model.backbone import *
from keras.layers import *
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy

class MyModel(Network):
    def __init__(self, args):
        super(MyModel, self).__init__(args)

    def build_model(self):
        inputs = Input((self.h, self.w, self.c))
        output = UNet_backbone(inputs, self.class_num)
        model = Model(inputs=inputs, outputs=output, name=self.model_name)
        self.model = model
        return model
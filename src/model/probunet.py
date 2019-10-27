# -*- coding: utf-8 -*-
import keras
from model.common import *
from model.backbone import *
from keras.layers import *
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
class MyModel(Network):
    def __init__(self, args):
        print('[Init ProbUNet]')
        self.fcomb = None
        self.beta = args.beta
        self.prior_net = None
        self.posterior_net = None
        super(MyModel, self).__init__(args)

    def build_model(self, use_mean=False):
        '''
         In this part, we want to let q(Z) closed to p(Z|X), which is the posterior distribution
         of hidden variables
        :param use_mean:
        :return:
        '''
        self.use_mean = use_mean
        self.inputs = Input((self.h, self.w, self.c))
        self.seg = Input((self.h, self.w, self.class_num))
        concat = Concatenate(axis=-1, name='concat')([self.inputs, self.seg])
        self.prior_parameter = AxisAlignedConvGaussian(self.inputs, name='prior_')
        self.posterior_parameter = AxisAlignedConvGaussian(concat, name='posterior_')
        self.unet_features = self.unet(self.inputs)
        if use_mean:
            self.func = lambda x: tfd.MultivariateNormalDiag(loc=x[0], scale_diag=tf.exp(x[1])).loc
        else:
            self.func = lambda x: tfd.MultivariateNormalDiag(loc=x[0], scale_diag=tf.exp(x[1])).sample()
        self.sample_logits = self.sample()
        self.rec_logits = self.reconstruct()

        self.train_model = Model(inputs=[self.inputs, self.seg], outputs=[self.rec_logits, self.prior_parameter[0],
                                                    self.prior_parameter[1],
                                                    self.posterior_parameter[0],
                                                    self.posterior_parameter[1]], name='posterior_pipeline')
        self.model = Model(inputs=self.inputs, outputs=[self.sample_logits, self.prior_parameter[0],
                                                   self.prior_parameter[1]], name='prior_pipeline')

        self.train_model.compile(loss={'logits_ac': self.elbo_loss(self.train_model.output)},
                                 metrics={'logits_ac': self.acc(self.train_model.output)}, optimizer=Adam(1e-5))
        self.model.summary()
        self.model.compile(loss=categorical_crossentropy,
                             metrics=[categorical_accuracy], optimizer=Adam(1e-5))



    def sample(self):
        z_p = Lambda(self.func, name='z_p')(self.prior_parameter)
        return FcombDecoder(self.unet_features, z_p, class_num=self.class_num)

    def reconstruct(self):
        z_q = Lambda(self.func, name='z_q')(self.posterior_parameter)
        return FcombDecoder(self.unet_features, z_q, class_num=self.class_num)


    def unet(self, inputs, name='Unet'):
        outputs = UNetBN_backbone(inputs, self.class_num, logits=True, name=name)
        return outputs

    def elbo_loss(self, output):
        def kl_loss(x):
            mu_p = x[0]
            log_p = x[1]
            mu_q = x[2]
            log_q = x[3]
            p = tfd.MultivariateNormalDiag(loc=mu_p, scale_diag=tf.exp(log_p))
            q = tfd.MultivariateNormalDiag(loc=mu_q, scale_diag=tf.exp(log_q))
            if self.use_mean:
                kl = tfd.kl_divergence(q, p)
            else:
                z_q = q.sample()
                log_q = q.log_prob(z_q)
                log_p = p.log_prob(z_q)
                kl = log_q - log_p
            return kl
        def elbo_func(y_true, y_pred):
            kl = tf.reduce_mean(Lambda(kl_loss)(output[1:]))
            ce = tf.reduce_mean(categorical_crossentropy(y_pred=output[0], y_true=y_true))
            return self.beta * kl + ce
        return elbo_func

    def acc(self, output):
        def acc_func(y_true, y_pred):
            seg = output[0]
            return categorical_accuracy(y_true=y_true, y_pred=seg)
        return acc_func

    def train(self, train_data, train_label, val_data, val_label):
        steps = int(train_data.shape[0] / self.args.batch_size)
        val_steps = int(val_data.shape[0] / self.args.batch_size)
        data_gen_args = dict(vertical_flip=True,
                             horizontal_flip=True)
        train_datagen = ProbGenerator(train_data, train_label, data_gen_args, self.args.batch_size)
        val_datagen = ProbGenerator(val_data, val_label, data_gen_args, self.args.batch_size)
        self.train_model.fit_generator(
            generator=train_datagen,
            epochs=self.args.epoch,
            steps_per_epoch=steps,
            validation_steps=val_steps,
            verbose=1,
            callbacks=self.callbacks,
            validation_data=val_datagen,
        )







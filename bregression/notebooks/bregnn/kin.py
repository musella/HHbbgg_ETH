import numpy as np

from keras.layers import Input, Dense, Add, Multiply
from keras.layers import Reshape, UpSampling1D, Flatten, Concatenate, Cropping1D, Convolution1D, RepeatVector
from keras.layers import Activation, LeakyReLU, PReLU, Lambda
from keras.layers import BatchNormalization, Dropout, GaussianNoise
from keras.models import Model, Sequential
from keras.layers import Layer
from keras.constraints import non_neg

import keras.optimizers

from keras.regularizers import l1,l2

from keras import backend as K

from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint

from sklearn.base import BaseEstimator

from . import losses 
from .ffwd import get_block

# --------------------------------------------------------------------------------------------------
def get_activations(activations,layers):
    if type(activations) == str:
        ret = [activations]*len(layers)
    elif len(activations) < len(layers):
        rat = len(layers) // len(activations)
        remind = len(layers) % len(activations)
        ret  = activations*rat
        if remind > 0:
            ret += activations[:remind]
    else:
        ret = activations
    return ret


# --------------------------------------------------------------------------------------------------
class KinRegression(BaseEstimator):

    def __init__(self,name,jets_shape,ev_shape,
                 linv_shape,
                 output_shape=None,
                 non_neg=False,
                 jets_layers=[32]*4,
                 jets_activations="lrelu",
                 ev_layers=[16]*4,
                 ev_activations="lrelu",
                 fc1_layers=[512,256,128],
                 fc1_activations="lrelu",
                 fc2_layers=[128,64,32],
                 fc2_activations="lrelu",
                 dropout=0.2, # 0.5 0.2
                 batch_norm=True,
                 do_bn0=True,
                 const_output_biases=None, 
                 noise=None,
                 optimizer="Adam", optimizer_params=dict(lr=1.e-3), # mse: 1e-3/5e-4
                 loss="RegularizedGaussNll",
                 loss_params=dict(),# dict(reg_sigma=3.e-2),
                 monitor_dir=".",
                 save_best_only=True,
                 valid_frac=None,
    ):
        self.name = name
        self.jets_shape = jets_shape
        self.ev_shape = ev_shape
        self.linv_shape = linv_shape
        
        self.output_shape = output_shape
        self.const_output_biases = const_output_biases

        self.non_neg = non_neg
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.use_bias = not batch_norm
        
        self.jets_layers = jets_layers
        self.jets_activations = get_activations(jets_activations,self.jets_layers)
        self.ev_layers = ev_layers
        self.ev_activations = get_activations(ev_activations,self.ev_layers)
        self.fc1_layers = fc1_layers
        self.fc1_activations = get_activations(fc1_activations,self.fc1_layers)
        if len(self.fc1_layers) > 0:
            self.fc1_layers.append( self.linv_shape[-1] )
            self.fc1_activations.append( None )
        self.fc2_layers = fc2_layers
        self.fc2_activations = get_activations(fc2_activations,self.fc2_layers)

        self.do_bn0 = do_bn0
        self.noise = noise
        
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.loss = loss
        self.loss_params = loss_params

        self.valid_frac = valid_frac
        self.save_best_only = save_best_only
        self.monitor_dir = monitor_dir
        
        self.model = None

        super(KinRegression,self).__init__()
        
    # ----------------------------------------------------------------------------------------------
    def __call__(self,docompile=False):
        
        if hasattr(losses,self.loss):
            loss = getattr(losses,self.loss)
            ## print(loss,isinstance(loss,object))
            if isinstance(loss,object):
                loss = loss(**self.loss_params)
            ## print(loss)
        else:
            loss = self.loss

        output_shape = self.output_shape
        if output_shape is None:
            output_shape = (getattr(loss,"n_params",1),)
            
        if self.model is None:
            jets_inputs = Input(shape=self.jets_shape,name="%s_jets_inp" % self.name)
            ev_inputs = Input(shape=self.ev_shape,name="%s_ev_inp" % self.name)
            linv_inputs = Input(shape=self.linv_shape,name="%s_linv_inp" % self.name)

            inputs = [jets_inputs,ev_inputs,linv_inputs]
            
            Ljets = jets_inputs
            Lev = ev_inputs
            Llinv = linv_inputs

            if self.do_bn0:
                Ljets = BatchNormalization(name="%s_jets_nb0" % self.name)(Ljets)
                Lev = BatchNormalization(name="%s_ev_nb0" % self.name)(Lev)
                Llinv = BatchNormalization(name="%s_linb_nb0" % self.name)(Llinv)
            Ljets0 = Ljets


            if len(self.ev_layers) > 0:
                Lev = get_block(Lev,self.name+"_ev",False,
                                self.batch_norm,self.noise,self.use_bias,self.dropout,
                                self.ev_layers,self.ev_activations)            
                
            def get1x1(*args,**kwargs):
                kwargs['kernel_size'] = 1
                return Convolution1D(*args,**kwargs)

            Levjets = RepeatVector(self.jets_shape[0],name="%s_ev_jets_rpt" % self.name)(Lev)
            Ljets = Concatenate(name="%s_jets_concat" % self.name)([Ljets,Levjets])
            Ljets = get_block(Ljets,self.name+"_jets",False,
                              self.batch_norm,self.noise,self.use_bias,self.dropout,
                              self.jets_layers,self.jets_activations,
                              core = get1x1 )
            Ljets = Flatten(name="%s_jets_flt" % self.name)(Ljets)
            Ljets0 = Flatten(name="%s_jets_inp_flt" % self.name)(Ljets0)
            Ljets = Concatenate(name="%s_jets_out_concat" % self.name)([Ljets,Ljets0])
            
            if len(self.fc1_layers)>0:
                Levj = Concatenate(name="%s_ev_jets_concat" % self.name)([Ljets,Lev])
                Levj = get_block(Levj,"%s_ev_jets" % self.name,False,
                                 self.batch_norm,self.noise,self.use_bias,self.dropout,
                                 self.fc1_layers,self.fc1_activations)
                
                L = Multiply(name="%s_mult" % self.name)([Levj,Llinv])
            else:
                L = Concatenate(name="%s_concat" % self.name)([Ljets,Lev,Llinv])
                
            L = get_block(L,self.name,False,
                         self.batch_norm,self.noise,self.use_bias,self.dropout,
                         self.fc2_layers,self.fc2_activations)            
                        
            reshape = False
            out_size = output_shape[0]
            if len(output_shape) > 1:
                reshape = True
                for idim in output_shape[1:]:
                    out_size *= output_shape[idim]
            constr = None
            output = Dense(out_size,use_bias=self.const_output_biases is None,
                           name="%s_out" % self.name)(L)
            if reshape:
                output = Reshape(output_shape,name="%s_rshp" % self.name)(output)
            if self.const_output_biases is not None:
                ## output = Lambda(lambda x: x+K.constant(self.const_output_biases))(output)
                output = ConstOffsetLayer(self.const_output_biases,name="%s_outb" % self.name)(output)

            if self.non_neg:
                output = Activation("relu",name="%s_outpos" % self.name)(output)
            self.model = Model( inputs=inputs, outputs=output )
            
        if docompile:
            optimizer = getattr(keras.optimizers,self.optimizer)(**self.optimizer_params)

            self.model.compile(optimizer=optimizer,loss=loss,metrics=[losses.mse0,losses.mae0,
                                                                      losses.r2_score0])
        return self.model

    # ----------------------------------------------------------------------------------------------
    def get_callbacks(self,has_valid=False,monitor='loss',save_best_only=True):
        if has_valid:
            monitor = 'val_'+monitor
        monitor_dir = self.monitor_dir
        csv = CSVLogger("%s/metrics.csv" % monitor_dir)
        checkpoint = ModelCheckpoint("%s/model-{epoch:02d}.hdf5" % monitor_dir,
                                     monitor=monitor,save_best_only=save_best_only,
                                     save_weights_only=False)
        return [csv,checkpoint]
    
    # ----------------------------------------------------------------------------------------------
    def fit(self,Xjets,Xev,Xlinv,y,**kwargs):

        model = self(True)
        
        has_valid = kwargs.get('validation_data',None) is not None
        if not has_valid and self.valid_frac is not None:
            last_train = int( Xjets.shape[0] * (1. - self.valid_frac) )
            Xjets_train = Xjets[:last_train]
            Xjets_valid = Xjets[last_train:]
            Xev_train = Xev[:last_train]
            Xev_valid = Xev[last_train:]
            Xlinv_train = Xlinv[:last_train]
            Xlinv_valid = Xlinv[last_train:]
            y_train = y[:last_train]
            y_valid = y[last_train:]
            kwargs['validation_data'] = ([Xjets_valid,Xev_valid,Xlinv_valid],y_valid)
            has_valid = True
        else:
            Xjets_train, Xev_valid, Xlinv_valid, y_train = Xjets, Xev, Xlinv, y
        X_train = [Xjets_train, Xev_valid, Xlinv_valid]
            
        if not 'callbacks' in kwargs:
            save_best_only=kwargs.pop('save_best_only',self.save_best_only)
            kwargs['callbacks'] = self.get_callbacks(has_valid=has_valid,
                                                     save_best_only=save_best_only)

        print(len(X_train))
        return model.fit(X_train,y_train,**kwargs)
    
    # ----------------------------------------------------------------------------------------------
    def predict(self,Xjets,Xev,Xlinv,p0=True,**kwargs):
        y_pred =  self.model.predict([Xjets,Xev,Xlinv],**kwargs)
        if p0:
            return y_pred[:,0]
        else:
            return y_pred
    
    # ----------------------------------------------------------------------------------------------
    def score(self,X,y,**kwargs):
        return -self.model.evaluate(X,y,**kwargs)
    

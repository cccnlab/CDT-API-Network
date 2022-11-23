import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet152


class MultiAttBlock(Model):
    def __init__(self):
        super(MultiAttBlock, self).__init__()
        self.multi_att = layers.MultiHeadAttention(num_heads=8, key_dim=256)
        self.add = layers.Add()
        self.norm = layers.LayerNormalization()
        
    def call(self, x):
        x_att = self.multi_att(x,x)
        x_att = self.add([x_att, x])
        x_att = self.norm(x_att)
       
        return x_att


class GateGen(Model):
    def __init__(self):
        super(GateGen, self).__init__()
        self.sig = layers.Activation("sigmoid")
        
    def call(self, x_1, x_2):
        
        h = tf.math.multiply(x_1,x_2)
       
        return self.sig(h)

class ResAtt(Model):
    def __init__(self):
        super(ResAtt, self).__init__()
        
    def call(self, x, g):
        
        a = tf.math.multiply(x,g)
       
        return tf.math.add(x,a)

class API(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        
        self.encode_dense = layers.Dense(512, activation="relu")
        self.dropout = layers.Dropout(0.1)
        self.decode_dense = layers.Dense(2048, activation="relu")
        
        self.concat = layers.Concatenate(axis=-1)
        
        self.gate = GateGen()
        
        self.res_att = ResAtt()
        
    
    def call(self, x_1, x_2, training=False):
        concat = self.concat([x_1, x_2])
        x = self.encode_dense(concat)
        
        if training:
            x = self.dropout(x, training=training)
            
        x_m = self.decode_dense(x)
        
        g_1 = self.gate(x_m,x_1)
        g_2 = self.gate(x_m,x_2)
            
        x_self_1 = self.res_att(x_1,g_1)
        x_other_1 = self.res_att(x_1,g_2)
        x_self_2 = self.res_att(x_2,g_2)
        x_other_2 = self.res_att(x_2,g_1)

        x_self_1 = self.dropout(x_self_1)
        x_other_1 = self.dropout(x_other_1)
        x_self_2 = self.dropout(x_self_2)
        x_other_2 = self.dropout(x_other_2)
        
        return x_self_1, x_other_1, x_self_2, x_other_2


class Classifier(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.imagenet_model = ResNet152(
                                input_shape=(256,256,3),
                                weights="imagenet",
                                include_top=False)
                                
        self.mid_layer = tf.keras.Model(self.imagenet_model.inputs,
                                        [self.imagenet_model.get_layer("conv5_block3_out").output, self.imagenet_model.output])
        
        
        self.avg_pool = layers.GlobalAveragePooling2D(name="avg_pool")
        self.api = API()
        self.dense = layers.Dense(3, name='dense_classifier')
        self.softmax_act = layers.Activation("softmax", dtype='float32', name='predictions')

    def set_imagenet_trainable(self, training):
        
        self.imagenet_model.trainable = training
        
    def reinitialize_prediction(self):
        
        self.dense = layers.Dense(3, name='dense_classifier')
        
    
    def get_embedding(self, inputs):
        
        embedding = self.imagenet_model(inputs)

        embedding = self.avg_pool(embedding)
        
        return embedding
    
    def train_pair(self, inputs_1, inputs_2):
        
        embedding_1 = self.imagenet_model(inputs_1)
        embedding_2 = self.imagenet_model(inputs_2)
        
        x_1 = self.avg_pool(embedding_1)
        x_2 = self.avg_pool(embedding_2)

        x_self_1, x_other_1, x_self_2, x_other_2 = self.api(x_1, x_2, training=True)
        
        p_self_1 = self.softmax_act(self.dense(x_self_1))
        p_other_1 = self.softmax_act(self.dense(x_other_1))
        p_self_2 = self.softmax_act(self.dense(x_self_2))
        p_other_2 = self.softmax_act(self.dense(x_other_2))
        
        return p_self_1, p_other_1, p_self_2, p_other_2
    
    
    def grad_cam(self, inputs):
        
        last_conv_embedding, embedding = self.mid_layer(inputs)

        x = self.avg_pool(embedding)


        pred = self.softmax_act(self.dense(x))
            
        return last_conv_embedding, pred
    
    def pred(self, inputs):
        embedding = self.imagenet_model(inputs)

        x = self.avg_pool(embedding)


        pred = self.softmax_act(self.dense(x))

        return pred
    
    
    
    def call(self, inputs_1, inputs_2):
        
        embedding_1 = self.imagenet_model(inputs_1)
        embedding_2 = self.imagenet_model(inputs_2)
        
        x_1 = self.avg_pool(embedding_1)
        x_2 = self.avg_pool(embedding_2)

        x_self_1, x_other_1, x_self_2, x_other_2 = self.api(x_1, x_2, training=True)
        
        p_self_1 = self.softmax_act(self.dense(x_self_1))
        p_other_1 = self.softmax_act(self.dense(x_other_1))
        p_self_2 = self.softmax_act(self.dense(x_self_2))
        p_other_2 = self.softmax_act(self.dense(x_other_2))
        
        return p_self_1, p_other_1, p_self_2, p_other_2

def score_rank_reg(p_other,p_self,y):
    index = tf.argmax(y, axis=-1)
    
    index = tf.reshape(index, (-1,1))
    index = tf.cast(index, tf.int32)
    
    idx = tf.stack([tf.reshape(tf.range(p_self.shape[0]), (-1,1)), index], axis=-1)
    
    p_other = tf.gather_nd(p_other, idx)
    p_self = tf.gather_nd(p_self, idx)
    
    
    diff = tf.math.subtract(p_other,p_self)
    diff = tf.math.add(diff,5e-3)
    
    return tf.reshape(tf.math.maximum(0.0,diff), [-1])




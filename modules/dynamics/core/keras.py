import tensorflow as tf
"""
Model object has loss
Net object has loss
Layer is very basic module that requires weights and shape definition
"""

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()



"""
network format should be
outputs =  call(inputs)
def loss(outputs, gts)
def summ(outputs, gts)

"""

class Net(tf.keras.Model):
    def __init__(self, name=None):
       super(Net, self).__init__(name=name)
       self.namescope = self.name
    def update_name_scope(self, p_name_scope):
       if p_name_scope != "":
        self.namescope = p_name_scope + "/" + self.namescope
    def remove_classname(self, ns):
        """
        remove the class name of a name space
        """
        return "/".join(ns.split("/")[1:])

    def summ(self):
        raise NotImplementedError
    def loss(self):
        raise NotImplementedError

class Layer(tf.keras.layers.Layer):
    def __init__(self):
       super(Layer, self).__init__()

import numpy as np
import numpy.matlib
import tensorflow as tf
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float64')
varType = 'float64'

# Create the neural network architecture
class myModel(tf.keras.models.Model):
    
    def __init__(self,c,nu,alpha,x0,xf,H):
        super(myModel,self).__init__()
        self.dense1 = tf.keras.layers.Dense(H,input_shape=(2,),activation='tanh')
        self.dense2 = tf.keras.layers.Dense(H,input_shape=(H,),activation='tanh')
        self.dense3 = tf.keras.layers.Dense(H,input_shape=(H,),activation='tanh')
        self.dense4 = tf.keras.layers.Dense(H,input_shape=(H,),activation='tanh')
        self.dense5 = tf.keras.layers.Dense(H,input_shape=(H,),activation='tanh')
        self.dense6 = tf.keras.layers.Dense(H,input_shape=(H,),activation='tanh')
        self.dense7 = tf.keras.layers.Dense(1,input_shape=(H,),activation='linear')

        self.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.Adam())
        self._setupBfgs = False

        self.c = tf.convert_to_tensor(c,dtype=varType)
        self.nu = tf.convert_to_tensor(nu,dtype=varType)
        self.alpha = tf.convert_to_tensor(alpha,dtype=varType)
        self.x0 = tf.convert_to_tensor(x0,dtype=varType)
        self.xf = tf.convert_to_tensor(xf,dtype=varType)
        self.two = tf.constant(2.,dtype='float64')

    def nn(self,x,t):
        dark = self.dense1(tf.concat([x,t],1))
        dark = self.dense2(dark)
        dark = self.dense3(dark)
        dark = self.dense4(dark)
        #dark = self.dense5(dark)
        #dark = self.dense6(dark)
        return self.dense7(dark)

    def ua(self,x,t):
        return self.c/self.alpha-self.c/self.alpha*tf.tanh(self.c*(x-self.c*t)/(self.two*self.nu))

    def u1(self,x,t):
        return self.nn(x,t)+(self.xf-x)/(self.xf-self.x0)*(self.ua(self.x0*tf.ones_like(x),t)-self.nn(self.x0*tf.ones_like(x),t))+x*(self.ua(self.xf*tf.ones_like(x),t)-self.nn(self.xf*tf.ones_like(x),t))
    def u(self,x,t):
        return self.u1(x,t)+self.ua(x,tf.zeros_like(t))-self.u1(x,tf.zeros_like(t))

    def call(self,inputs,training=False):
        x = tf.convert_to_tensor(inputs[:,0:1])
        t = tf.convert_to_tensor(inputs[:,1:2])
        if training:
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                g.watch(t)

                u = self.u(x,t)

                dudt = g.gradient(u,t)
                dudx = g.gradient(u,x)
                d2udx2 = g.gradient(dudx,x)

            del g
            return (dudt+self.alpha*u*dudx-self.nu*d2udx2)**2

        else:
            return self.u(x,t)

    def trainBfgs(self,inputs,outputs,maxIter=100):
        if not self._setupBfgs:
            self.call(inputs)
            self.lossFun = tf.keras.losses.MeanSquaredError()
            self._setupBfgs = True
        self.func = self.function_factory(self.lossFun,inputs,outputs)
        initParams = tf.dynamic_stitch(self.func.idx,self.trainable_variables)
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=self.func, initial_position=initParams, max_iterations=maxIter)
        self.func.assign_new_model_parameters(results.position)

    def function_factory(self, loss, train_x, train_y):
        """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
        Args:
            self [in]: an instance of `tf.keras.Model` or its subclasses.
            loss [in]: a function with signature loss_value = loss(pred_y, true_y).
            train_x [in]: the input part of training data.
            train_y [in]: the output part of training data.
        Returns:
            A function that has a signature of:
                loss_value, gradients = f(model_parameters).
        """

        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.trainable_variables)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = [] # stitch indices
        part = [] # partition indices

        for i, shape in enumerate(shapes):
            n = numpy.product(shape)
            idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.
            Args:
                params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
            """

            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.trainable_variables[i].assign(tf.reshape(param, shape))

        # now create a function that will be returned by this factory
        @tf.function
        def f(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.
            This function is created by function_factory.
            Args:
               params_1d [in]: a 1D tf.Tensor.
            Returns:
                A scalar loss and the gradients w.r.t. the `params_1d`.
            """

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss_value = loss(self(train_x, training=True), train_y)

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value, self.trainable_variables)
            grads = [k if k is not None else tf.zeros_like(self.trainable_variables[i]) for i,k in enumerate(grads)]
            grads = tf.dynamic_stitch(idx,grads)

            # print out iteration & loss
            f.iter.assign_add(1)
            tf.print("Iter:", f.iter, "loss:", loss_value, "\t", end='\r')

            return loss_value, grads

        # store these information as members so we can use them outside the scope
        f.iter = tf.Variable(0)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters

        return f



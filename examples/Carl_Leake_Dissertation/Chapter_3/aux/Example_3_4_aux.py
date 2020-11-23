import numpy as np
import numpy.matlib
import tensorflow as tf
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float64')

# Create the neural network architecture
class myModel(tf.keras.models.Model):
    
    def __init__(self,H):
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
        self.one = tf.constant(1.,dtype='float64')
        self.two = tf.constant(2.,dtype='float64')
        self.six = tf.constant(6.,dtype='float64')

    def nn(self,x,y):
        dark = self.dense1(tf.concat([x,y],1))
        dark = self.dense2(dark)
        dark = self.dense3(dark)
        dark = self.dense4(dark)
        dark = self.dense5(dark)
        dark = self.dense6(dark)
        return self.dense7(dark)

    def u1(self,x,y):
        return self.nn(x,y)+(self.one-x)*(y**3-self.nn(tf.zeros_like(x),y))+x*((self.one+y**3)*tf.exp(-self.one)-self.nn(tf.ones_like(x),y))
    def u(self,x,y):
        return self.u1(x,y)+(self.one-y)*(x*tf.exp(-x)-self.u1(x,tf.zeros_like(y)))+y*((x+self.one)*tf.exp(-x)-self.u1(x,tf.ones_like(y)))

    def call(self,inputs,training=False):
        x = tf.convert_to_tensor(inputs[:,0:1])
        y = tf.convert_to_tensor(inputs[:,1:2])

        if training:
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                g.watch(y)
                u = self.u(x,y)
                d2udx2 = g.gradient(g.gradient(u,x),x)
                d2udy2 = g.gradient(g.gradient(u,y),y)
            return d2udx2+d2udy2-tf.exp(-x)*(x-self.two+y**3+self.six*y)
        else:
            return self.u(x,y)

    def trainBfgs(self,inputs,outputs,maxIter=100,tol=1e-10):
        if not self._setupBfgs:
            self.predict(inputs)
            self.lossFun = tf.keras.losses.MeanSquaredError()
            self._setupBfgs = True
        self.func = self.function_factory(self.lossFun,inputs,outputs)
        initParams = tf.dynamic_stitch(self.func.idx,self.trainable_variables)
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=self.func, initial_position=initParams, max_iterations=maxIter, tolerance=tol)
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
            tf.print("Iter:", f.iter, "loss:", loss_value, end='\r')

            return loss_value, grads

        # store these information as members so we can use them outside the scope
        f.iter = tf.Variable(0)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters

        return f



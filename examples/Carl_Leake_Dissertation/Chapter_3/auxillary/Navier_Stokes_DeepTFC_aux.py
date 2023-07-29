import numpy as np
import numpy.matlib
import tensorflow as tf
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float64')
varType = 'float64'

# Create the neural network architecture
class myModel(tf.keras.models.Model):
    
    def __init__(self,rho,h,L,dPdx,mu,H):
        super(myModel,self).__init__()
        self.dense1 = tf.keras.layers.Dense(H,input_shape=(3,),activation='tanh')
        self.dense2 = tf.keras.layers.Dense(H,input_shape=(H,),activation='tanh')
        self.dense3 = tf.keras.layers.Dense(H,input_shape=(H,),activation='tanh')
        self.dense4 = tf.keras.layers.Dense(H,input_shape=(H,),activation='tanh')
        self.dense5 = tf.keras.layers.Dense(2,input_shape=(H,),activation='linear')

        self.compile(loss=tf.keras.losses.MeanAbsoluteError(),optimizer=tf.keras.optimizers.Adam())
        self._setupBfgs = False

        self.rho = tf.convert_to_tensor(rho,dtype=varType)
        self.H = tf.convert_to_tensor(h,dtype=varType)
        self.L = tf.convert_to_tensor(L,dtype=varType)
        self.dPdx = tf.convert_to_tensor(dPdx,dtype=varType)
        self.mu = tf.convert_to_tensor(mu,dtype=varType)

    def nn(self,x,y,t):
        dark = self.dense1(tf.concat([x,y,t],1))
        dark = self.dense2(dark)
        dark = self.dense3(dark)
        dark = self.dense4(dark)
        dark = self.dense5(dark)
        return dark[:,0:1],dark[:,1:2]

    def nx(self,x,y,t):
        with tf.GradientTape(persistent=True) as gx:
            gx.watch(x)
            dark = self.dense1(tf.concat([x,y,t],1))
            dark = self.dense2(dark)
            dark = self.dense3(dark)
            dark = self.dense4(dark)
            dark = self.dense5(dark)
            dark1 = gx.gradient(dark[:,0:1],x)
            dark2 = gx.gradient(dark[:,1:2],x)
        del gx
        return dark1,dark2

    def call(self,inputs,training=False):
        # Note that this ce is written out in expanded form rather than in recursive form because it
        # speeds up the model noticably. This is true whenever we have more than one dependent variable
        # coming off of the same neural network. 

        one = tf.constant(1.,dtype='float64')
        two = tf.constant(2.,dtype='float64')
        L = tf.constant(self.L,dtype='float64')
        x = tf.convert_to_tensor(inputs[:,0:1])
        y = tf.convert_to_tensor(inputs[:,1:2])
        t = tf.convert_to_tensor(inputs[:,2:3])
        if training:
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                g.watch(y)
                g.watch(t)
                zerosx = tf.zeros_like(x)
                Lx = tf.ones_like(x)*L
                zerost = tf.zeros_like(t)
                h2my = -tf.ones_like(y)*self.H/two
                h2py = tf.ones_like(y)*self.H/two
                nnu,nnv = self.nn(x,y,t)
                nnxy0u,nnxy0v = self.nn(x,y,zerost)
                nn0y0u,nn0y0v = self.nn(zerosx,y,zerost)
                nn0ytu,nn0ytv = self.nn(zerosx,y,t)
                nx0y0u,nx0y0v = self.nx(Lx,y,zerost)
                nx0ytu,nx0ytv = self.nx(Lx,y,t)
                nn0h2m0u,nn0h2m0v = self.nn(zerosx,h2my,zerost)
                nnxh2m0u,nnxh2m0v = self.nn(x,h2my,zerost)
                nx0h2m0u,nx0h2m0v = self.nx(Lx,h2my,zerost)
                nn0h2mtu,nn0h2mtv = self.nn(zerosx,h2my,t)
                nnxh2mtu,nnxh2mtv = self.nn(x,h2my,t)
                nx0h2mtu,nx0h2mtv = self.nx(Lx,h2my,t)
                nn0h2p0u,nn0h2p0v = self.nn(zerosx,h2py,zerost)
                nnxh2p0u,nnxh2p0v = self.nn(x,h2py,zerost)
                nx0h2p0u,nx0h2p0v = self.nx(Lx,h2py,zerost)
                nn0h2ptu,nn0h2ptv = self.nn(zerosx,h2py,t)
                nnxh2ptu,nnxh2ptv = self.nn(x,h2py,t)
                nx0h2ptu,nx0h2ptv = self.nx(Lx,h2py,t)

                u = nnu-nnxy0u+nn0y0u-nn0ytu+x*nx0y0u-x*nx0ytu+one/(two*self.H)*((two*y-self.H)*(nn0h2m0u-nnxh2m0u+x*nx0h2m0u-nn0h2mtu+nnxh2mtu-x*nx0h2mtu)-(self.H+two*y)*(nn0h2p0u-nnxh2p0u+x*nx0h2p0u-nn0h2ptu+nnxh2ptu-x*nx0h2ptu))
                v = nnv-nnxy0v+nn0y0v-nn0ytv+x*nx0y0v-x*nx0ytv+one/(two*self.H)*((two*y-self.H)*(nn0h2m0v-nnxh2m0v+x*nx0h2m0v-nn0h2mtv+nnxh2mtv-x*nx0h2mtv)-(self.H+two*y)*(nn0h2p0v-nnxh2p0v+x*nx0h2p0v-nn0h2ptv+nnxh2ptv-x*nx0h2ptv))

                dudx = g.gradient(u,x)
                dvdy = g.gradient(v,y)

                dudt = g.gradient(u,t)
                dudy = g.gradient(u,y)
                d2udx2 = g.gradient(dudx,x)
                d2udy2 = g.gradient(dudy,y)

                dvdt = g.gradient(v,t)
                dvdx = g.gradient(v,x)
                d2vdx2 = g.gradient(dvdx,x)
                d2vdy2 = g.gradient(dvdy,y)
                
            del g
            return (dudx+dvdy)**2+(self.rho*(dudt+u*dudx+v*dudy)+self.dPdx-self.mu*(d2udx2+d2udy2))**2+(self.rho*(dvdt+u*dvdx+v*dvdy)-self.mu*(d2vdx2+d2vdy2))**2

        else:
            zerosx = tf.zeros_like(x)
            Lx = tf.ones_like(x)*L
            zerost = tf.zeros_like(t)
            h2my = -tf.ones_like(y)*self.H/two
            h2py = tf.ones_like(y)*self.H/two
            nnu,nnv = self.nn(x,y,t)
            nnxy0u,nnxy0v = self.nn(x,y,zerost)
            nn0y0u,nn0y0v = self.nn(zerosx,y,zerost)
            nn0ytu,nn0ytv = self.nn(zerosx,y,t)
            nx0y0u,nx0y0v = self.nx(Lx,y,zerost)
            nx0ytu,nx0ytv = self.nx(Lx,y,t)
            nn0h2m0u,nn0h2m0v = self.nn(zerosx,h2my,zerost)
            nnxh2m0u,nnxh2m0v = self.nn(x,h2my,zerost)
            nx0h2m0u,nx0h2m0v = self.nx(Lx,h2my,zerost)
            nn0h2mtu,nn0h2mtv = self.nn(zerosx,h2my,t)
            nnxh2mtu,nnxh2mtv = self.nn(x,h2my,t)
            nx0h2mtu,nx0h2mtv = self.nx(Lx,h2my,t)
            nn0h2p0u,nn0h2p0v = self.nn(zerosx,h2py,zerost)
            nnxh2p0u,nnxh2p0v = self.nn(x,h2py,zerost)
            nx0h2p0u,nx0h2p0v = self.nx(Lx,h2py,zerost)
            nn0h2ptu,nn0h2ptv = self.nn(zerosx,h2py,t)
            nnxh2ptu,nnxh2ptv = self.nn(x,h2py,t)
            nx0h2ptu,nx0h2ptv = self.nx(Lx,h2py,t)

            u = nnu-nnxy0u+nn0y0u-nn0ytu+x*nx0y0u-x*nx0ytu+one/(two*self.H)*((two*y-self.H)*(nn0h2m0u-nnxh2m0u+x*nx0h2m0u-nn0h2mtu+nnxh2mtu-x*nx0h2mtu)-(self.H+two*y)*(nn0h2p0u-nnxh2p0u+x*nx0h2p0u-nn0h2ptu+nnxh2ptu-x*nx0h2ptu))
            v = nnv-nnxy0v+nn0y0v-nn0ytv+x*nx0y0v-x*nx0ytv+one/(two*self.H)*((two*y-self.H)*(nn0h2m0v-nnxh2m0v+x*nx0h2m0v-nn0h2mtv+nnxh2mtv-x*nx0h2mtv)-(self.H+two*y)*(nn0h2p0v-nnxh2p0v+x*nx0h2p0v-nn0h2ptv+nnxh2ptv-x*nx0h2ptv))
            return u,v

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



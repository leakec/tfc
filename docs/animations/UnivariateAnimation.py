from jax import jit
import jax.numpy as np
from tfc.utils import MakePlot

import matplotlib
#COLOR = [0.,]*3
COLOR = [0.467,]*3
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR
matplotlib.rcParams['axes.edgecolor'] = COLOR

# u(0) = y(3)
# u(1) = 1
# u(2) = -1

n = 150
x = np.linspace(0.,3.,100)
a = np.linspace(1.,2.,n)
b = np.linspace(2.,8.,n)
c = {'a':a[-1],'b':b[-1]}

g = lambda x,c: c['a']*x*np.sin(c['b']*x)
u = jit(lambda x,c: g(x,c)+\
        (-x**3+7.*x-6.)/6.*(g(x[-1],c)-g(x[0],c))+\
        (x**3-9.*x+10.)/2.*(1.-g(np.array(1.),c))+\
        (-x**3+9.*x-8.)/2.*(-1.-g(np.array(2.),c)))


U = u(x,c)
p = MakePlot(r'$x$',r'$u(x,g(x))$')
p.ax[0].set_ylim([-10.,30.])
rel3 = p.ax[0].plot([0.,3.],[U[0],U[0]],'r',linestyle='--')[0]
plot = p.ax[0].plot(x,U,color=COLOR,label=r'Constrained expression')[0]
rel1 = p.ax[0].plot(0.,U[0],'r',linestyle='--',marker='.',markersize=10,label=r'Relative constraint $u(0) = u(3)$')[0]
rel2 = p.ax[0].plot(3.,U[0],'r',linestyle='None',marker='.',markersize=10)[0]
p.ax[0].plot(1.,1.,'b',linestyle='None',marker='.',markersize=10,label=r'Absolute constraint $u(1) = 1$')
p.ax[0].plot(2.,-1.,'c',linestyle='None',marker='.',markersize=10,label=r'Absolute constraint $u(2) = -1$')
legend = p.ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.00),
          ncol=1, fancybox=True, shadow=False)
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((0, 0, 1, 0.0))
p.fig.subplots_adjust(top=0.76)
p.ax[0].grid(True)
p.PartScreen(8,7)
p.show()

def anim():
    for k in range(n):
        c.update({'a':a[k],'b':b[k]})
        U = u(x,c)
        plot.set_ydata(U)
        rel1.set_ydata(U[0])
        rel2.set_ydata(U[0])
        rel3.set_ydata([U[0],U[0]])
        yield
    for k in range(n,0,-1):
        c.update({'a':a[k],'b':b[k]})
        U = u(x,c)
        plot.set_ydata(U)
        rel1.set_ydata(U[0])
        rel2.set_ydata(U[0])
        rel3.set_ydata([U[0],U[0]])
        yield

p.animate(anim,save=False)

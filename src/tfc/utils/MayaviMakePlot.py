import numpy as np
from mayavi import mlab
from matplotlib import colors as mcolors

def _hex_to_rgb(val):
    val = val.lstrip('#')
    return tuple(int(val[i:i+2],16) for i in range(0,6,2))

colors = dict((key,_hex_to_rgb(mcolors.CSS4_COLORS[key])) for key in mcolors.CSS4_COLORS)
colors.update(mcolors.BASE_COLORS)

class MakePlot:

    _colors = colors

    @staticmethod
    def _ProcessKwargs(**kwargs):
        """ This function effectively extends common mlab keywords. """
        # Process color argument if it exists
        if 'color' in kwargs:
            if kwargs['color'] in MakePlot._colors:
                kwargs['color'] = MakePlot._colors[kwargs['color']]

        return kwargs

    @staticmethod
    def ColormapGradient(c1,c2):
        if isinstance(c1,str) and c1 in MakePlot._colors:
            c1 = np.array([*MakePlot._colors[c1],1.])
        if isinstance(c2,str) and c2 in MakePlot._colors:
            c2 = np.array([*MakePlot._colors[c2],1.])
        c1 = c1*255
        c2 = c2*255
        return np.linspace(c1,c2,256,dtype=np.uint8)

    @staticmethod
    def SetColormapGradient(obj,c1,c2):
        obj.module_manager.scalar_lut_manager.lut.table = MakePlot.ColormapGradient(c1,c2)

    def SetColormapGradient(self,obj,c1,c2):
        obj.module_manager.scalar_lut_manager.lut.table = MakePlot.ColormapGradient(c1,c2)
                
    def __init__(self):
        self.fig = mlab.figure(bgcolor=(1.,1.,1.))
        self.scene = self.fig.scene

    def FullScreen(self):
        """ This function makes the plot fullscreen. """

        # Get screensize
        import tkinter as tk
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        # Get dpi and set new figsize
        dpi = float(self.fig.get_dpi())
        mlab.figure(self.fig,size=(width/dpi,height/dpi))

    def PartScreen(self,width,height):
        """ This function makes the plot width x height inches. """
        mlab.figure(self.fig,size=(width,height))

    def animate(self,animFunc,outDir='MyMovie',fileName='images',save=True):
        if save:
            import os
            if not os.path.exists(outDir):
                os.mkdir(outDir)
            k = 0

            iterable = animFunc()
            while next(iterable,-1) != -1:
                fileNameFull = '{}{:0>6d}'.format(fileName,k)
                self.save(os.path.join(outDir,fileNameFull),fileType='png')
                k += 1

            print("ffmpeg -r 60 -i ./{0}/{1}%06d.png -c:v libx264 -profile:v high -pix_fmt yuv420p ./{0}/MyMovie.mp4".format(outDir,fileName))
        else:
            mlab.animate(func=animFunc,delay=10,ui=True,support_movie=False)()

    def save(self,fileName,fileType='pdf'):
        mlab.savefig(fileName+'.'+fileType,figure=self.fig)

    def show(self):
        """ Re-draw the class's figure. """
        return mlab.draw(figure=self.fig)

    @property
    def show_axes(self):
        """ Axes indicator. """
        return self.scene.show_axes

    @show_axes.setter
    def show_axes(self,val):
        """ Set axes indicator. """
        self.scene.show_axes = val

    def view(self,*args,**kwargs):
        """ Call mlab's view on the class's figure. """
        return mlab.view(*args,figure=self.fig,**kwargs)

    def points3d(self,*args,**kwargs):
        """ Call mlab's points3d on the class's figure. """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.points3d(*args,figure=self.fig,**kwargs)

    def plot3d(self,*args,**kwargs):
        """ Call mlab's plot3d on the class's figure. """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.plot3d(*args,figure=self.fig,**kwargs)

    def surf(self,*args,**kwargs):
        """ Call mlab's surf on the class's figure. """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.surf(*args,figure=self.fig,**kwargs)

    def quiver3d(self,*args,**kwargs):
        """ Call mlab's quiver3d on the class's figure. """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.quiver3d(*args,figure=self.fig,**kwargs)

    def mesh(self,*args,**kwargs):
        """ Call mlab's mesh on the class's figure. """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.mesh(*args,figure=self.fig,**kwargs)

    def plot3d(self,*args,**kwargs):
        """ Call mlab's plot3d on the class's figure. """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.plot3d(*args,figure=self.fig,**kwargs)

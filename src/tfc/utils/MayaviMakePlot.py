import numpy as np
from mayavi import mlab
from matplotlib import colors as mcolors


class MakePlot:
    """MakePlot class for Mayavi."""

    @staticmethod
    def _str_to_rgb(color):
        """Call matplotlib's colorConverter.to_rgb on input string.

        Parameters
        ----------
        color : str
            Color string


        Returns
        -------
        color_rgb : tuple
            3-tuple of the RGB for the color
        """
        return mcolors.colorConverter.to_rgb(color)

    @staticmethod
    def _str_to_rgba(color, alpha=None):
        """Call matplotlib's colorConverter.to_rgba on input string.

        Parameters
        ----------
        color : str
            Color string

        alpha : float
            Alpha value to use in the return RGBA. If None, then the returned alpha = 1. (Default value = None)

        Returns
        -------
        color_rgba : tuple
            4-tuple of the RGB for the color
        """
        return mcolors.colorConverter.to_rgba(color, alpha=None)

    @staticmethod
    def _ProcessKwargs(**kwargs):
        """This function effectively extends common mlab keywords.

        Parameters
        ----------
        **kwargs : dict
            keyword arguments

        Returns
        -------
        kwargs : dict
            Same as input keyword arguments but color has been transformed to an RGB if it was a string.
        """
        # Process color argument if it exists
        if "color" in kwargs:
            if isinstance(kwargs["color"], str):
                kwargs["color"] = MakePlot._str_to_rgb(kwargs["color"])

        return kwargs

    @staticmethod
    def ColormapGradient(c1, c2):
        """Returns a Mayavi LUT for a gradient that linearly transforms from c1 to c2.

        Parameters
        ----------
        c1 : str or list
            Input color as a string or 4-element RGBA list.

        c2 : str or list
            Input color as a string or 4-element RGBA list.

        Returns
        -------
        lut : array-like
            Mayavi LUT for a gradient that linearly transforms from c1 to c2.

        """
        if isinstance(c1, str):
            c1 = np.array(MakePlot._str_to_rgba(c1))
        if isinstance(c2, str):
            c2 = np.array(MakePlot._str_to_rgba(c2))
        c1 = c1 * 255
        c2 = c2 * 255
        return np.linspace(c1, c2, 256, dtype=np.uint8)

    @staticmethod
    def SetColormapGradient(obj, c1, c2):
        """Applies a linear, gradient colormap to the object. The colors in the gradient are c1 and c2.

        Parameters
        ----------
        obj : mlab object
            mlab object to apply the colormap gradient to.

        c1 : str or list
            Input color as a string or 4-element RGBA list.

        c2 : str or list
            Input color as a string or 4-element RGBA list.
        """
        obj.module_manager.scalar_lut_manager.lut.table = MakePlot.ColormapGradient(c1, c2)

    def __init__(self):
        self.fig = mlab.figure(bgcolor=(1.0, 1.0, 1.0))
        self.scene = self.fig.scene

    def FullScreen(self):
        """This function makes the plot fullscreen."""

        # Get screensize
        import tkinter as tk

        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        # Get dpi and set new figsize
        dpi = float(self.fig.get_dpi())
        mlab.figure(self.fig, size=(width / dpi, height / dpi))

    def PartScreen(self, width, height):
        """This function makes the plot width x height inches.

        Parameters
        ----------
        width : float
            Width of the plot in inches.

        height : float
            Height of the plot in inches.
        """
        mlab.figure(self.fig, size=(width, height))

    def animate(self, animFunc, outDir="MyMovie", fileName="images", save=True, delay=10):
        """

        Parameters
        ----------
        animFunc : function generator
            Function that modifies what is displayed on the plot

        outDir : str, optional
             Directory to save frames in: only used if save = True. (Default value = "MyMovie")

        fileName : str, optional
             Filename for the frames: only used if save = True. (Default value = "images")

        save : bool, optional
             Controls whether the function saves the frames of the animation or not. (Default value = True)

        delay: int in [10,100000], optional
             Controls the delay used by mlab.animate. (Default value = 10)

        Returns
        -------
        animator : mayavi.tools.animator.Animator
            Returns an animator object only if save = False. Otherwise, there is no return value.
        """
        if save:
            import os

            if not os.path.exists(outDir):
                os.mkdir(outDir)
            k = 0

            iterable = animFunc()
            while next(iterable, -1) != -1:
                fileNameFull = "{}{:0>6d}".format(fileName, k)
                self.save(os.path.join(outDir, fileNameFull), fileType="png")
                k += 1

            print(
                "ffmpeg -r 60 -i ./{0}/{1}%06d.png -c:v libx264 -profile:v high -pix_fmt yuv420p ./{0}/MyMovie.mp4".format(
                    outDir, fileName
                )
            )
        else:
            return mlab.animate(func=animFunc, delay=delay)()

    def save(self, fileName, fileType="pdf"):
        """This function saves the figure.

        Parameters
        ----------
        fileName : str
            Filename where the figure should be saved. Note, this should not include the file extension.

        fileType : str, optional
            File exension to use. (Default value = "pdf")
        """
        mlab.savefig(fileName + "." + fileType, figure=self.fig)

    def show(self):
        """Re-draw the class's figure."""
        return mlab.draw(figure=self.fig)

    @property
    def show_axes(self):
        """Axes indicator."""
        return self.scene.show_axes

    @show_axes.setter
    def show_axes(self, val):
        """Set axes indicator.

        Parameters
        ----------
        val : bool
            True turn on the axes indicator, and False turns off the axes indicator.
        """
        self.scene.show_axes = val

    def view(self, *args, **kwargs):
        """Call mlab's view on the class's figure.

        Parameters
        ----------
        *args : iterable
            args passed on to view

        **kwargs : dict
            kwargs passed on to view
        """
        return mlab.view(*args, figure=self.fig, **kwargs)

    def points3d(self, *args, **kwargs):
        """Call mlab's points3d on the class's figure.

        Parameters
        ----------
        *args : iterable
            args passed on to points3d

        **kwargs : dict
            kwargs passed on to points3d after being processed by _ProcessKwargs.
        """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.points3d(*args, figure=self.fig, **kwargs)

    def plot3d(self, *args, **kwargs):
        """Call mlab's plot3d on the class's figure.

        Parameters
        ----------
        *args : iterable
            args passed on to plot3d

        **kwargs : dict
            kwargs passed on to plot3d after being processed by _ProcessKwargs.
        """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.plot3d(*args, figure=self.fig, **kwargs)

    def surf(self, *args, **kwargs):
        """Call mlab's surf on the class's figure.

        Parameters
        ----------
        *args : iterable
            args passed on to surf

        **kwargs : dict
            kwargs passed on to surf after being processed by _ProcessKwargs.
        """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.surf(*args, figure=self.fig, **kwargs)

    def quiver3d(self, *args, **kwargs):
        """Call mlab's quiver3d on the class's figure.

        Parameters
        ----------
        *args : iterable
            args passed on to quiver3d

        **kwargs : dict
            kwargs passed on to quiver3d after being processed by _ProcessKwargs.
        """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.quiver3d(*args, figure=self.fig, **kwargs)

    def mesh(self, *args, **kwargs):
        """Call mlab's mesh on the class's figure.

        Parameters
        ----------
        *args : iterable
            args passed on to mesh

        **kwargs : dict
            kwargs passed on to mesh after being processed by _ProcessKwargs.
        """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.mesh(*args, figure=self.fig, **kwargs)

    def contour3d(self, *args, **kwargs):
        """Call mlab's contour3d on the class's figure.

        Parameters
        ----------
        *args : iterable
            args passed on to contour3d

        **kwargs : dict
            kwargs passed on to mesh after being processed by _ProcessKwargs.
        """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.contour3d(*args, figure=self.fig, **kwargs)

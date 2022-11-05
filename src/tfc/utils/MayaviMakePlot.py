import numpy as np
import numpy.typing as npt
import mayavi
from mayavi import mlab
from matplotlib import colors as mcolors
from .types import Dict, Tuple, Path, Ge, Le, Annotated, Literal
from typing import Optional, Any, Union, Generator, Callable
from .TFCUtils import TFCPrint

Color = Union[str, Tuple[float, float, float, float], npt.NDArray[np.float64]]
TFCPrint()


class MakePlot:
    """MakePlot class for Mayavi."""

    @staticmethod
    def _str_to_rgb(color: str) -> Tuple[float, float, float]:
        """Call matplotlib's colorConverter.to_rgb on input string.

        Parameters
        ----------
        color : str
            Color string

        Returns
        -------
        color_rgb : Tuple[float, float, float]
            3-tuple of the RGB for the color
        """
        return mcolors.colorConverter.to_rgb(color)

    @staticmethod
    def _str_to_rgba(color, alpha: Optional[float] = None) -> Tuple[float, float, float, float]:
        """Call matplotlib's colorConverter.to_rgba on input string.

        Parameters
        ----------
        color : str
            Color string

        alpha : float
            Alpha value to use in the return RGBA. If None, then the returned alpha = 1. (Default value = None)

        Returns
        -------
        color_rgba : Tuple[float, float, float, float]
            4-tuple of the RGB for the color
        """
        return mcolors.colorConverter.to_rgba(color, alpha=alpha)

    @staticmethod
    def _ProcessKwargs(**kwargs: Any) -> Dict[str, Any]:
        """This function effectively extends common mlab keywords.

        Parameters
        ----------
        **kwargs : Any
            keyword arguments

        Returns
        -------
        kwargs : Dict[str, any]
            Same as input keyword arguments but color has been transformed to an RGB if it was a string.
        """
        # Process color argument if it exists
        if "color" in kwargs:
            if isinstance(kwargs["color"], str):
                kwargs["color"] = MakePlot._str_to_rgb(kwargs["color"])

        return kwargs

    @staticmethod
    def ColormapGradient(c1: Color, c2: Color) -> npt.NDArray[np.uint8]:
        """Returns a Mayavi LUT for a gradient that linearly transforms from c1 to c2.

        Parameters
        ----------
        c1 : Color
            Input color as a string or 4-element RGBA list.

        c2 : Color
            Input color as a string or 4-element RGBA list.

        Returns
        -------
        lut : npt.NDArray[np.uint8]
            Mayavi LUT for a gradient that linearly transforms from c1 to c2.
        """
        if isinstance(c1, str):
            c1 = np.array(MakePlot._str_to_rgba(c1))
        if isinstance(c2, str):
            c2 = np.array(MakePlot._str_to_rgba(c2))
        if not isinstance(c1, np.ndarray):
            c1 = np.array(c1)
        if not isinstance(c2, np.ndarray):
            c2 = np.array(c2)
        c1 = c1 * 255
        c2 = c2 * 255
        return np.linspace(c1, c2, 256, dtype=np.uint8)

    @staticmethod
    def SetColormapGradient(obj: Any, c1: Color, c2: Color):
        """
        Applies a linear, gradient colormap to the object. The colors in the gradient are c1 and c2.

        Parameters
        ----------
        obj : Any
            mlab object to apply the colormap gradient to.
        c1 : Color
            Input color as a string or 4-element RGBA list.
        c2 : Color
            Input color as a string or 4-element RGBA list.
        """
        obj.module_manager.scalar_lut_manager.lut.table = MakePlot.ColormapGradient(c1, c2)

    def __init__(self):
        """
        This function initializes the plot.
        """

        self.fig = mlab.figure(bgcolor=(1.0, 1.0, 1.0))
        self.scene = self.fig.scene

    def FullScreen(self):
        """
        This function makes the plot fullscreen.
        """

        # Get screensize
        import tkinter as tk

        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        # Get dpi and set new figsize
        dpi = float(self.fig.get_dpi())
        mlab.figure(self.fig, size=(width / dpi, height / dpi))

    def PartScreen(self, width: float, height: float):
        """This function makes the plot width x height inches.

        Parameters
        ----------
        width : float
            Width of the plot in inches.

        height : float
            Height of the plot in inches.
        """
        mlab.figure(self.fig, size=(width, height))

    IntRange = Annotated[int, Ge(10), Le(100000)]

    def animate(
        self,
        animFunc: Callable[[], Generator[None, None, None]],
        outDir: Path = "MyMovie",
        fileName: str = "images",
        save: bool = True,
        delay: IntRange = 10,
    ) -> Optional[mayavi.tools.animator.Animator]:
        """

        Parameters
        ----------
        animFunc : Callable[[], Generator[None, None, None]]
            Function that modifies what is displayed on the plot
        outDir : Path, optional
            Directory to save frames in: only used if save = True. (Default value = "MyMovie")
        fileName : str, optional
            Filename for the frames: only used if save = True. (Default value = "images")
        save : bool, optional
            Controls whether the function saves the frames of the animation or not. (Default value = True)
        delay: IntRange in [10,100000], optional
            Controls the delay used by mlab.animate. (Default value = 10)

        Returns
        -------
        animator : Optional[mayavi.tools.animator.Animator]
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

    def save(
        self,
        fileName: Path,
        fileType: Literal[
            "png",
            "jpg",
            "bmp",
            "tiff",
            "ps",
            "eps",
            "pdf",
            "rib",
            "oogl",
            "iv",
            "wrl",
            "vrml",
            "obj",
            "x3d",
            "pov",
            None,
        ] = None,
    ):
        """
        This function saves the figure.

        Parameters
        ----------
        fileName : Path
            Filename where the figure should be saved.
        fileType : Literal["png", "jpg", "bmp", "tiff", "ps", "eps", "pdf", "rib", "oogl", "iv", "wrl", "vrml", "obj", "x3d", "pov", None], optional
            File suffix to use. If None, then the suffix will be inferred from the file name. (Default value = None)
        """
        if not fileType:
            from pathlib import Path

            suffix = Path(fileName).suffix[1:]
            if suffix in [
                "png",
                "jpg",
                "bmp",
                "tiff",
                "ps",
                "eps",
                "pdf",
                "rib",
                "oogl",
                "iv",
                "wrl",
                "vrml",
                "obj",
                "x3d",
                "pov",
            ]:
                fileType = suffix
            else:
                fileType = "pdf"
                TFCPrint.Warning(
                    f"Warning, file type could not be inferred from {fileName}. The file type has been set to pdf."
                )
                fileName += "." + fileType
        else:
            fileName += "." + fileType
        mlab.savefig(fileName, figure=self.fig)

    def show(self):
        """Re-draw the class's figure."""
        return mlab.draw(figure=self.fig)

    @property
    def show_axes(self):
        """
        Adds axes indicator.
        """
        return self.scene.show_axes

    @show_axes.setter
    def show_axes(self, val: bool):
        """
        Set axes indicator.

        Parameters
        ----------
        val : bool
            True turns on the axes indicator and False turns off the axes indicator.
        """
        self.scene.show_axes = val

    def view(self, *args: Any, **kwargs: Any):
        """
        Call mlab's view on the class's figure.

        Parameters
        ----------
        *args : Any
            args passed on to view
        **kwargs : Any
            kwargs passed on to view
        """
        return mlab.view(*args, figure=self.fig, **kwargs)

    def points3d(self, *args: Any, **kwargs: Any) -> mayavi.modules.glyph.Glyph:
        """
        Call mlab's points3d on the class's figure.

        Parameters
        ----------
        *args : Any
            args passed on to points3d
        **kwargs : Any
            kwargs passed on to points3d after being processed by _ProcessKwargs.

        Returns:
        --------
        points3d : mayavi.modules.glyph.Glyph
            Glyphs corresponding to the points.
        """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.points3d(*args, figure=self.fig, **kwargs)

    def plot3d(self, *args: Any, **kwargs: Any) -> mayavi.modules.surface.Surface:
        """Call mlab's plot3d on the class's figure.

        Parameters
        ----------
        *args : Any
            args passed on to plot3d
        **kwargs : Any
            kwargs passed on to plot3d after being processed by _ProcessKwargs.

        Returns:
        --------
        surf : mayavi.modules.surface.Surface
            Line as a Mayavi surface.
        """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.plot3d(*args, figure=self.fig, **kwargs)

    def surf(self, *args: Any, **kwargs: Any) -> mayavi.modules.surface.Surface:
        """Call mlab's surf on the class's figure.

        Parameters
        ----------
        *args : Any
            args passed on to surf
        **kwargs : Any
            kwargs passed on to surf after being processed by _ProcessKwargs.

        Returns:
        --------
        surf : mayavi.modules.surface.Surface
            Surface.
        """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.surf(*args, figure=self.fig, **kwargs)

    def quiver3d(self, *args: Any, **kwargs: Any) -> mayavi.modules.vectors.Vectors:
        """
        Call mlab's quiver3d on the class's figure.

        Parameters
        ----------
        *args : Any
            args passed on to quiver3d
        **kwargs : Any
            kwargs passed on to quiver3d after being processed by _ProcessKwargs.

        Returns:
        --------
        vectors : mayavi.modules.vectors.Vectors
            Vectors associated with the quiver3d plot.
        """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.quiver3d(*args, figure=self.fig, **kwargs)

    def mesh(self, *args: Any, **kwargs: Any) -> mayavi.modules.surface.Surface:
        """Call mlab's mesh on the class's figure.

        Parameters
        ----------
        *args : Any
            args passed on to mesh
        **kwargs : Any
            kwargs passed on to mesh after being processed by _ProcessKwargs.

        Returns:
        --------
        mesh : mayavi.modules.surface.Surface
            Mesh as a mayavi surface.
        """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.mesh(*args, figure=self.fig, **kwargs)

    def contour3d(self, *args: Any, **kwargs: Any) -> mayavi.modules.iso_surface.IsoSurface:
        """
        Call mlab's contour3d on the class's figure.

        Parameters
        ----------
        *args : Any
            args passed on to contour3d
        **kwargs : Any
            kwargs passed on to mesh after being processed by _ProcessKwargs.

        Returns:
        --------
        contour : mayavi.modules.iso_surface.IsoSurface
            Coutour.
        """
        kwargs = MakePlot._ProcessKwargs(**kwargs)
        return mlab.contour3d(*args, figure=self.fig, **kwargs)

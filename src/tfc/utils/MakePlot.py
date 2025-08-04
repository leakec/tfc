import os
import pickle
import numpy as np

import matplotlib.pyplot as plt

from .TFCUtils import TFCPrint
from .tfc_types import StrArrayLike, Path, Literal, pint
from typing import Optional, Generator, Callable

TFCPrint()


class MakePlot:
    """
    This class is used to easily create journal-article-ready plots and subplots. The class can create 2D as well as 3D plots
    and even has support for twin y-axes.
    """

    def __init__(
        self,
        xlabs: StrArrayLike,
        ylabs: StrArrayLike,
        titles: Optional[StrArrayLike] = None,
        twinYlabs: Optional[StrArrayLike] = None,
        zlabs: Optional[StrArrayLike] = None,
        style: Optional[str | dict | Path | list[str] | list[dict] | list[Path]] = None,
    ):
        """
        This function initializes the plot/subplots based on the inputs provided.

        Parameters
        ----------
        xlabs: StrArrayLike
            The x-axes labels for the plots
        ylabs: StrArrayLike
            The y-axes labels for the plots
        titles: StrArrayLike, optional
            The titles for the plots. (Default value = None)
        twinYlabs: StrArrayLike, optional
            The twin y-axes labels for the plots. Setting this forces twin axis y-axes. (Default value = None)
        zlabs: StrArrayLike, optional
            The z-axes labels of for the plots. Setting this forces subplots to be 3D. (Default value = None)
        style : str| dict| Path| list[str]| list[dict]| list[Path]
            Matplotlib style. (Default value = None)
        """

        # Apply a style if specified
        if style:
            plt.style.use(style)

        # Set the fontsizes and family
        smallSize = 16
        mediumSize = 18
        largeSize = 18
        plt.rc("font", size=smallSize)
        plt.rc("axes", titlesize=mediumSize)
        plt.rc("axes", labelsize=largeSize)
        plt.rc("xtick", labelsize=mediumSize)
        plt.rc("ytick", labelsize=mediumSize)
        plt.rc("legend", fontsize=smallSize)
        plt.rc("figure", titlesize=largeSize)

        # Create figure and store basic labels
        self.fig = plt.figure()

        # Consistify all label types
        if isinstance(xlabs, np.ndarray):
            pass
        elif isinstance(xlabs, str):
            xlabs = np.array([[xlabs]])
        elif isinstance(xlabs, tuple) or isinstance(xlabs, list):
            xlabs = np.array(xlabs)
        else:
            TFCPrint.Error(
                "The xlabels provided are not of a valid type. Please provide valid xlabels"
            )
        if len(xlabs.shape) == 1:
            xlabs = np.expand_dims(xlabs, 1)

        if isinstance(ylabs, np.ndarray):
            pass
        elif isinstance(ylabs, str):
            ylabs = np.array([[ylabs]])
        elif isinstance(ylabs, tuple) or isinstance(ylabs, list):
            ylabs = np.array(ylabs)
        else:
            TFCPrint.Error(
                "The ylabels provided are not of a valid type. Please provide valid ylabels"
            )
        if len(ylabs.shape) == 1:
            ylabs = np.expand_dims(ylabs, 1)

        if not zlabs is None:
            if isinstance(zlabs, np.ndarray):
                pass
            elif isinstance(zlabs, str):
                zlabs = np.array([[zlabs]])
            elif isinstance(zlabs, tuple) or isinstance(zlabs, list):
                zlabs = np.array(zlabs)
            else:
                TFCPrint.Error(
                    "The zlabels provided are not of a valid type. Please provide valid zlabels"
                )
            if len(zlabs.shape) == 1:
                zlabs = np.expand_dims(zlabs, 1)

        if titles is not None:
            if isinstance(titles, np.ndarray):
                pass
            elif isinstance(titles, str):
                titles = np.array([[titles]])
            elif isinstance(titles, tuple) or isinstance(titles, list):
                titles = np.array(titles)
            else:
                TFCPrint.Error(
                    "The titles provided are not of a valid type. Please provide valid titles."
                )
            if len(titles.shape) == 1:
                titles = np.expand_dims(titles, 1)

        if twinYlabs is not None:
            if isinstance(twinYlabs, np.ndarray):
                pass
            elif isinstance(twinYlabs, str):
                twinYlabs = np.array([[twinYlabs]])
            elif isinstance(twinYlabs, tuple) or isinstance(twinYlabs, list):
                twinYlabs = np.array(twinYlabs)
            else:
                TFCPrint.Error(
                    "The twin ylabels provided are not of a valid type. Please provide valid twin ylabels"
                )
            if len(twinYlabs.shape) == 1:
                twinYlabs = np.expand_dims(twinYlabs, 1)

        # Create all subplots and add labels
        if zlabs is None:
            n = xlabs.shape
            self.ax = list()
            count = 0
            for j in range(n[0]):
                for k in range(n[1]):
                    if xlabs[j, k] is None:
                        continue
                    self.ax.append(self.fig.add_subplot(n[0], n[1], j * n[1] + k + 1))
                    self.ax[count].set_xlabel(xlabs[j, k])
                    self.ax[count].set_ylabel(ylabs[j, k])
                    count += 1
        else:
            n = xlabs.shape
            self.ax = list()
            count = 0
            for j in range(n[0]):
                for k in range(n[1]):
                    if xlabs[j, k] is None:
                        continue
                    self.ax.append(
                        self.fig.add_subplot(n[0], n[1], j * n[1] + k + 1, projection="3d")
                    )
                    self.ax[count].set_xlabel(xlabs[j, k])
                    self.ax[count].set_ylabel(ylabs[j, k])
                    self.ax[count].set_zlabel(zlabs[j, k])
                    count += 1

        if twinYlabs is not None:
            self.twinAx = list()
            count = 0
            for j in range(n[0]):
                for k in range(n[1]):
                    if xlabs[j, k] is None:
                        continue
                    self.twinAx.append(self.ax[count].twinx())
                    self.twinAx[count].set_ylabel(twinYlabs[j, k])
                    count += 1

        # Add titles if desired
        if titles is not None:
            count = 0
            for j in range(n[0]):
                for k in range(n[1]):
                    if titles[j, k] is None:
                        continue
                    self.ax[count].set_title(titles[j, k])
                    count += 1

        # Set tight layout for the figure
        self.fig.tight_layout()

    def FullScreen(self):
        """
        This function makes the plot fullscreen.
        """

        # Get screensize
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        # Get dpi and set new figsize
        dpi = float(self.fig.get_dpi())
        self.fig.set_size_inches(width / dpi, height / dpi)

    def PartScreen(self, width: float, height: float):
        """
        This function makes the plot width x height inches.

        Parameters
        ----------
        width : float
            Width of the plot in inches.

        height : float
            Height of the plot in inches.
        """
        self.fig.set_size_inches(width, height)

    def show(self):
        """
        This function shows the plot.
        """
        self.fig.show()

    def draw(self):
        """
        This function draws the canvas.
        """
        self.fig.canvas.draw()

    def save(
        self,
        fileName: Path,
        transparent: bool = True,
        fileType: Literal["png", "pdf", "ps", "eps", "svg", None] = None,
    ):
        """
        This function crops and saves the figure.

        Parameters
        ----------
        fileName : Path
            Filename where the figure should be saved.
        transparent : bool, optional
            Whether to save the plot with transparency or not. (Default value = True)
        fileType : Literal["png", "pdf", "ps", "eps", "svg", None], optional
            File suffix to use. If None, then the suffix will inferred from the suffix of the fileName. (Default value = None)
        """
        if not fileType:
            from pathlib import Path

            suffix = Path(fileName).suffix[1:]
            if suffix in ["png", "pdf", "ps", "eps", "svg"]:
                fileType = suffix
            else:
                fileType = "pdf"
                TFCPrint.Warning(
                    f"Warning, file type could not be inferred from {fileName}. The file type has been set to pdf."
                )
        else:
            fileName += "." + fileType

        self.fig.savefig(
            fileName,
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=300,
            format=fileType,
            transparent=transparent,
        )

    def savePickle(self, fileName: Path):
        """This function saves the figure in a pickle format so it can be opened and modified later.

        Parameters
        ----------
        fileName : Path
            Filename where the figure should be saved. Note, this should not include the file suffix.
        """
        pickle.dump(self, open(fileName + ".pickle", "wb"))

    def saveAll(
        self,
        fileName: Path,
        transparent: bool = True,
        fileType: Literal["png", "pdf", "ps", "eps", "svg", None] = None,
    ):
        """This function invokes the save and savePickle functions.

        Parameters
        ----------
        fileName : Path
            Filename where the figure should be saved.
        transparent : bool, optional
            Whether to save the plot with transparency or not. (Default value = True)
        fileType : Literal["png", "pdf", "ps", "eps", "svg", None], optional
            File suffix to use. If None, then the suffix will be inferred from the suffix of the fileName. (Default value = None)
        """
        self.save(fileName, transparent=transparent, fileType=fileType)
        self.savePickle(fileName)

    def animate(
        self,
        animFunc: Callable[[], Generator[None, None, None]],
        outDir: Path = "MyMovie",
        fileName: str = "images",
        save: bool = True,
        delay: pint = 10,
    ):
        """
        Creates an animation using a Python generator.

        Parameters
        ----------
        animFunc: Callable[[], Generator[None, None, None]]
            Function whose returned generator modifies what is displayed on the plot
        outDir : Path, optional
             Directory to save frames in: only used if save = True. (Default value = "MyMovie")
        fileName : str, optional
             Filename for the frames: only used if save = True. (Default value = "images")
        save : bool, optional
             Controls whether the function saves the frames of the animation or not. (Default value = True)
        delay : pint, optional
             Delay in milliseconds between frames: only used if save = False. (Default value = 10)
        """

        iterable = animFunc()

        if save:
            if not os.path.exists(outDir):
                os.mkdir(outDir)
            k = 0

            while next(iterable, -1) != -1:
                plt.pause(delay / 1000.0)
                fileNameFull = "{}{:0>6d}".format(fileName, k)
                self.save(os.path.join(outDir, fileNameFull), fileType="png")
                k += 1

            print(
                'ffmpeg -r 60 -i ./{0}/{1}%06d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ./{0}/MyMovie.mp4'.format(
                    outDir, fileName
                )
            )
        else:
            while next(iterable, -1) != -1:
                plt.pause(delay / 1000.0)

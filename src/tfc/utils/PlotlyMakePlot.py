import numpy as np
import plotly.graph_objects as go

from .TFCUtils import TFCPrint


class MakePlot:
    """
    A MakePlot class for Plotly.
    This class wraps common Plotly functions to ease figure creation.

    Parameters
    ----------
    xlabs: list or array-like
        The x-axes labels of for the plots

    ylabs: list or array-like
        The y-axes labels of for the plots

    zlabs: list or array-like, optional
        The z-axes labels of for the plots. Setting this forces subplots to be 3D. (Default value = None)

    titles: list or array-like, optional
        The titles for the plots. (Default value = None)
    """

    _gridColor = "rgb(176, 176, 176)"
    _fontSize = 16

    def __init__(self, xlabs, ylabs, titles=None, zlabs=None):

        # View distance used by the view method
        self.viewDistance = 2.5

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

        if zlabs is None:
            self._is3d = False
        else:
            self._is3d = True
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
                titles = titles.flatten().tolist()
            elif isinstance(titles, str):
                titles = [titles]
            elif isinstance(titles, tuple) or isinstance(titles, list):
                titles = np.array(titles).flatten().tolist()
            else:
                TFCPrint.Error(
                    "The titles provided are not of a valid type. Please provide valid titles."
                )

        if int(np.prod(xlabs.shape)) != 1:
            from plotly.subplots import make_subplots

            if self._is3d:
                specs = [
                    [
                        {"is_3d": True},
                    ]
                    * xlabs.shape[1],
                ] * xlabs.shape[0]
                self.fig = make_subplots(
                    rows=xlabs.shape[0], cols=xlabs.shape[1], specs=specs, subplot_titles=titles
                )
            else:
                self.fig = make_subplots(
                    rows=xlabs.shape[0], cols=xlabs.shape[1], subplot_titles=titles
                )
            self._hasSubplots = True
        else:
            self.fig = go.Figure()
            self._hasSubplots = False

        if self._hasSubplots:
            for row in range(xlabs.shape[0]):
                for col in range(xlabs.shape[1]):
                    if self._is3d:
                        if row == 0 and col == 0:
                            self.fig["layout"]["scene"]["xaxis"]["title"] = xlabs[row, col]
                            self.fig["layout"]["scene"]["yaxis"]["title"] = ylabs[row, col]
                            self.fig["layout"]["scene"]["zaxis"]["title"] = zlabs[row, col]
                        else:
                            self.fig["layout"]["scene" + str(row + col + 1)]["xaxis"][
                                "title"
                            ] = xlabs[row, col]
                            self.fig["layout"]["scene" + str(row + col + 1)]["yaxis"][
                                "title"
                            ] = ylabs[row, col]
                            self.fig["layout"]["scene" + str(row + col + 1)]["zaxis"][
                                "title"
                            ] = zlabs[row, col]
                    else:
                        if row == 0 and col == 0:
                            self.fig["layout"]["xaxis"]["title"] = xlabs[row, col]
                            self.fig["layout"]["yaxis"]["title"] = ylabs[row, col]
                        else:
                            self.fig["layout"]["xaxis" + str(row + col + 1)]["title"] = xlabs[
                                row, col
                            ]
                            self.fig["layout"]["yaxis" + str(row + col + 1)]["title"] = ylabs[
                                row, col
                            ]

        else:
            if self._is3d:
                self.fig.update_layout(scene=dict(xaxis=dict(title=xlabs[0, 0])))
                self.fig.update_layout(scene=dict(yaxis=dict(title=ylabs[0, 0])))
                self.fig.update_layout(scene=dict(zaxis=dict(title=zlabs[0, 0])))
            else:
                self.fig.update_xaxes(title=xlabs[0, 0])
                self.fig.update_yaxes(title=ylabs[0, 0])

        # Update grid and background colors
        if self._is3d:
            for row in range(xlabs.shape[0]):
                for col in range(xlabs.shape[1]):
                    if row == 0 and col == 0:
                        self.fig["layout"]["scene"]["xaxis"]["gridcolor"] = self._gridColor
                        self.fig["layout"]["scene"]["yaxis"]["gridcolor"] = self._gridColor
                        self.fig["layout"]["scene"]["zaxis"]["gridcolor"] = self._gridColor
                        self.fig["layout"]["scene"]["xaxis"]["backgroundcolor"] = "white"
                        self.fig["layout"]["scene"]["yaxis"]["backgroundcolor"] = "white"
                        self.fig["layout"]["scene"]["zaxis"]["backgroundcolor"] = "white"
                    else:
                        self.fig["layout"]["scene" + str(row + col + 1)]["xaxis"][
                            "gridcolor"
                        ] = self._gridColor
                        self.fig["layout"]["scene" + str(row + col + 1)]["yaxis"][
                            "gridcolor"
                        ] = self._gridColor
                        self.fig["layout"]["scene" + str(row + col + 1)]["zaxis"][
                            "gridcolor"
                        ] = self._gridColor
                        self.fig["layout"]["scene" + str(row + col + 1)]["xaxis"][
                            "backgroundcolor"
                        ] = "white"
                        self.fig["layout"]["scene" + str(row + col + 1)]["yaxis"][
                            "backgroundcolor"
                        ] = "white"
                        self.fig["layout"]["scene" + str(row + col + 1)]["zaxis"][
                            "backgroundcolor"
                        ] = "white"
        else:
            self.fig.update_xaxes(gridcolor=self._gridColor, linecolor="black")
            self.fig.update_yaxes(gridcolor=self._gridColor, linecolor="black")
            self.fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")

        # Update layout
        self.fig.update_layout(
            margin=dict(t=50, b=50, r=50, l=50), autosize=True, font=dict(size=MakePlot._fontSize)
        )

    def Surface(self, row=None, col=None, **kwargs):
        """
        Creates a plotly surface on the subplot specified by row and col or on
        the main figure if not using subplots.

        Parameters
        ----------
        row : int, optional
             subplot row (Default value = None)
        col : int, optional
             subplot column (Default value = None)
        **kwargs : dict, optional
            keyword arguments passed on to plotly.graphic_objects.Surface

        Returns
        -------
        surf : plotly.graphic_objects.Surface
        """
        return self.fig.add_trace(go.Surface(**kwargs), row=row, col=col)

    def Scatter3d(self, row=None, col=None, **kwargs):
        """
        Creates a 3d plotly scatter on the subplot specified by row and col or on
        the main figure if not using subplots.

        Parameters
        ----------
        row : int, optional
             subplot row (Default value = None)
        col : int, optional
             subplot column (Default value = None)
        **kwargs : dict, optional
            keyword arguments passed on to plotly.graphic_objects.Scatter3d

        Returns
        -------
        scatter : plotly.graphic_objects.Scatter3d
        """
        return self.fig.add_trace(go.Scatter3d(**kwargs), row=row, col=col)

    def Scatter(self, row=None, col=None, **kwargs):
        """
        Creates a plotly scatter on the subplot specified by row and col or on
        the main figure if not using subplots.

        Parameters
        ----------
        row : int, optional
             subplot row (Default value = None)
        col : int, optional
             subplot column (Default value = None)
        **kwargs : dict, optional
            keyword arguments passed on to plotly.graphic_objects.Scatter

        Returns
        -------
        scatter : plotly.graphic_objects.Scatter
        """
        return self.fig.add_trace(go.Scatter(**kwargs), row=row, col=col)

    def Histogram(self, row=None, col=None, **kwargs):
        """
        Creates a plotly histogram on the subplot specified by row and col or on
        the main figure if not using subplots.

        Parameters
        ----------
        row : int, optional
             subplot row (Default value = None)
        col : int, optional
             subplot column (Default value = None)
        **kwargs : dict, optional
            keyword arguments passed on to plotly.graphic_objects.Histogram

        Returns
        -------
        hist : plotly.graphic_objects.Histogram
        """
        return self.fig.add_trace(go.Histogram(**kwargs), row=row, col=col)

    def Contour(self, row=None, col=None, **kwargs):
        """
        Creates a plotly contour on the subplot specified by row and col or on
        the main figure if not using subplots.

        Parameters
        ----------
        row : int, optional
             subplot row (Default value = None)
        col : int, optional
             subplot column (Default value = None)
        **kwargs : dict, optional
            keyword arguments passed on to plotly.graphic_objects.Contour

        Returns
        -------
        contour : plotly.graphic_objects.Contour
        """
        return self.fig.add_trace(go.Contour(**kwargs), row=row, col=col)

    def Box(self, row=None, col=None, **kwargs):
        """
        Creates a plotly box on the subplot specified by row and col or on
        the main figure if not using subplots.

        Parameters
        ----------
        row : int, optional
             subplot row (Default value = None)
        col : int, optional
             subplot column (Default value = None)
        **kwargs : dict, optional
            keyword arguments passed on to plotly.graphic_objects.Box

        Returns
        -------
        box : plotly.graphic_objects.Box
        """
        return self.fig.add_trace(go.Box(**kwargs), row=row, col=col)

    def Violin(self, row=None, col=None, **kwargs):
        """
        Creates a plotly violin on the subplot specified by row and col or on
        the main figure if not using subplots.

        Parameters
        ----------
        row : int
             subplot row (Default value = None)
        col : int
             subplot column (Default value = None)
        **kwargs : dict, optional
            keyword arguments passed on to plotly.graphic_objects.Violin

        Returns
        -------
        violin : plotly.graphic_objects.Violin
        """
        return self.fig.add_trace(go.Violin(**kwargs), row=row, col=col)

    def show(self, **kwargs):
        """
        Calls the figures show method.

        Parameters
        ----------
        **kwargs : dict, optional
            keyword arguments passed on to fig.show
        """
        self.fig.show(**kwargs)

    def save(self, fileName, tight=True, fileType="pdf", **kwargs):
        """
        Saves the figure using the type specified. If HTML is specified, the figure will
        be saved as a dynamic html file. All other file types are static.

        Parameters
        ----------
        fileName : str
            Name of the save file minus the file type (e.g., MyFigure not MyFigure.pdf).

        tight : boolean, optional
            If the fileType is pdf and this value is true, then pdfCropMargins is used to eliminate whitespace.
            (Default value = True)
        fileType : {"pdf","jpg","png","html"}, optional
            Type of file to save the figure as. (Default value = "pdf")
        **kwargs : dict, optional
            Keyword arguments passed onto fig.write_image or fig.write_html, depending on fileType.
        """
        fileNameFull = fileName + "." + fileType
        if fileType == "html":
            self.fig.write_html(fileNameFull, **kwargs)
        else:
            self.fig.write_image(fileNameFull, **kwargs)
            if fileType == "pdf" and tight:
                from pdfCropMargins import crop

                crop(["-p", "0", fileNameFull])

    def UploadToPlotly(self, username, apiKey, fileName, autoOpen=False):
        """
        Upload your plot to Plotly.

        Parameters
        ----------
        username : str
            Plotly username

        apiKey : str
            Plotly api_key

        fileName : str
            Name of the file to save the plot as.

        autoOpen : bool, optional
            If true, plot will open in browser after saving. (Default value = False)
        """

        import chart_studio

        chart_studio.tools.set_credentials_file(username=username, api_key=apiKey)
        return chart_studio.plotly.plot(self.fig, filename=fileName, auto_open=autoOpen)

    def view(self, azimuth, elevation, row=None, col=None, viewDistance=None):
        """
        Change the view on the subplot specified by row and col or on
        the main figure if not using subplots.

        Parameters
        ----------
        azimuth : float
            Azimuth value in degrees
        elevation : float
            Elevation value in degrees

        row : int, optional
            subplot row (Default value = None)
        col : int, optional
            subplot column (Default value = None)
        viewDistance : float, optional
            Distance from camera to plot. (Default value = self.viewDistance)
        """

        if not self._is3d:
            TFCPrint.Error("The view method is only for 3d plots.")

        if viewDistance:
            self.viewDistance = viewDistance

        azimuth *= np.pi / 180.0
        elevation *= np.pi / 180.0
        dark = self.viewDistance * np.array(
            [
                -np.sin(azimuth) * np.cos(elevation),
                -np.cos(azimuth) * np.cos(elevation),
                np.sin(elevation),
            ]
        )
        if row and col:
            sceneNum = row + col - 2
            if sceneNum == 0:
                sceneNum = ""
            else:
                sceneNum = str(sceneNum)

            self.fig["layout"]["scene" + sceneNum]["camera"].eye = dict(
                x=dark[0], y=dark[1], z=dark[2]
            )
        else:
            self.fig["layout"]["scene"]["camera"].eye = dict(x=dark[0], y=dark[1], z=dark[2])

import os
from graphviz import Digraph
from yattag import Doc, indent
from .types import List
from .types import Path

class HTML:
    """
    This contains helper functions for creating HTML files with yattag.

    Parameters
    ----------
    outFile: str
        Output file
    """

    def __init__(self, outFile: Path):
        """
        This function initializes the header file, and saves useful variables to self.

        Parameters
        ----------
        outFile : Path
            Output file
        """
        self._outFile = outFile
        self.doc, self.tag, self.text = Doc().tagtext()
        self.centerClass = (
            ".center {\n\tdisplay: block;\n\tmargin-left: auto;\n\tmargin-right: auto;\n}\n"
        )

    def GenerateHtml(self) -> str:
        """
        This function generates and formats the HTML file text.

        Returns:
        --------
        html : str
            HTML file as a string.
        """

        html = indent(self.doc.getvalue(), indentation="", newline="\n")
        return html

    def WriteFile(self):
        """This function writes the HTML file text to a file."""
        if not os.path.exists(os.path.dirname(self._outFile)):
            os.makedirs(os.path.dirname(self._outFile))
        out = open(self._outFile, "w")
        out.write(self.GenerateHtml())
        out.close()

    def ReadFile(self, inFile: Path) -> str:
        """This function reads the file specified by "inFile" and retuns the
        contents as a string.

        Parameters
        ----------
        inFile : Path
            File to read.

        Returns
        -------
        outStr : str
            Contents of inFile as a string.
        """
        tmpFile = open(inFile, "r")
        dark = tmpFile.read()
        tmpFile.close()
        return dark


class Dot:
    """
    This class contains helper functions used to create dot graphs.
    """

    def __init__(self, outFile: Path, name: str):
        """
        This function initializes the class and creates the digraph.

        Parameters
        ----------
        outFile : Path
            Name of the filename under which the dot file should be saved.

        name : str
            What the dot file should be called by Digraph.
        """

        self._outFile = outFile
        self._name = name
        self.dot = Digraph(name=self._name)

    def Render(self, formats : List[str] = ["cmapx", "svg"]):
        """
        This function renders the dot graph as a .svg and as a .cmapx.

        Parameters
        ----------
        formats : List[str], optional
            List whose elementts dictate which formats to render the dot graph in. Default value = ["cmapx", "svg"]
        """
        for f in formats:
            self.dot.render(self._outFile, format=f, cleanup=True, view=False)

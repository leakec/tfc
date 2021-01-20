import numpy as np


class table:
    """This class is used to create the text needed for latex tables."""

    @staticmethod
    def _Header(numCols):
        """This function creates the table header based on the number of columns.

        Parameters
        ----------
        numCols : int
            Number of columns in the table

        Returns
        -------
        strOut : str
            Table header
        """
        return "\\begin{center}\n\\begin{tabular}{" + "|c" * numCols + "|}\n"

    @staticmethod
    def _colHeader(strIn):
        """This function creates the column header based on the list of strings that are passed in
        via the input strIn.

        Parameters
        ----------
        strIn : list
            List of strings that form the column headers.

        Returns
        -------
        strOut : str
            Column header
        """
        return " & ".join(strIn) + "\\\\\n"

    @staticmethod
    def _Arr2Tab(arrIn, form="%.4E", rowHeader=None):
        """This function transforms the 2-D numpy array (arrIn) into latex tabular format. The
        "form" argument specifies the number format to be used in the tabular environment.
        The "rowHeader" argument is a list of strings that are used in the first column of
        each row in the tabular environment. The latex tabular environment is returned as a
        string.

        Parameters
        ----------
        arrIn : array-like
            Array to convert to Latex table.

        form : str, optional
             Format string for the table numbers. (Default value = "%.4E")

        rowHeader : list, optional
             List of strings to use as the row headers. (Default value = None)

        Returns
        -------
        strOut : str
            Array formatted as a latex table.
        """
        out = str()
        if rowHeader is None:
            if np.size(arrIn.shape) == 2:
                numRows = arrIn.shape[0]
                for k in range(numRows):
                    out += np.array2string(
                        arrIn[k, :], separator=" & ", formatter={"float_kind": lambda x: form % x}
                    ).strip("[]")
                    out += "\\\\\n\\hline\n"
            else:
                out += np.array2string(
                    arrIn[k, :], separator=" & ", formatter={"float_kind": lambda x: form % x}
                ).strip("[]")
                out += "\\\\\n"
        else:
            if np.size(arrIn.shape) == 2:
                numRows = arrIn.shape[0]
                for k in range(numRows):
                    out += rowHeader[k] + " & "
                    out += np.array2string(
                        arrIn[k, :], separator=" & ", formatter={"float_kind": lambda x: form % x}
                    ).strip("[]")
                    out += "\\\\\n\\hline\n"
            else:
                out += np.array2string(
                    arrIn[k, :], separator=" & ", formatter={"float_kind": lambda x: form % x}
                ).strip("[]")
                out += "\\\\\n"
        return out.rstrip()

    @staticmethod
    def _Footer():
        """This function creates the footer for the latex table."""
        return "\\end{tabular}\n\\end{center}"

    @staticmethod
    def SimpleTable(arrIn, form="%.4E", colHeader=None, rowHeader=None):
        """This function creates a simple latex table for the 2D numpy array arrIn. The
        "form" argument specifies the number format to be used in the tabular environment.
        The "colHeader" arugment is a list of strings that are used as the first row in the
        tabular environment. The "rowHeader" argument is a list of strings that are used
        in the first column of each row in the tabular environment. The latex tabular
        environment is returned as a string.

        Parameters
        ----------
        arrIn : array-like
            Array to convert to Latex table.

        form : str, optional
             Format string for the table numbers. (Default value = "%.4E")

        colHeader : list, optional
            List of strings that form the column headers. (Default value = None)

        rowHeader : list, optional
             List of strings to use as the row headers. (Default value = None)

        Returns
        -------
            table : str
                Latex table as a string.
        """

        if colHeader is None and rowHeader is None:
            return (
                table._Header(arrIn.shape[1])
                + "\\hline\n"
                + table._Arr2Tab(arrIn, form=form)
                + "\n"
                + table._Footer()
            )
        elif rowHeader is None:
            return (
                table._Header(arrIn.shape[1])
                + "\\hline\n"
                + table._colHeader(colHeader)
                + "\\hline\n"
                + table._Arr2Tab(arrIn, form=form)
                + "\n"
                + table._Footer()
            )
        elif colHeader is None:
            return (
                table._Header(arrIn.shape[1] + 1)
                + "\\hline\n"
                + table._Arr2Tab(arrIn, form=form, rowHeader=rowHeader)
                + "\n"
                + table._Footer()
            )
        else:
            return (
                table._Header(arrIn.shape[1] + 1)
                + "\\hline\n"
                + table._colHeader(colHeader)
                + "\\hline\n"
                + table._Arr2Tab(arrIn, form=form, rowHeader=rowHeader)
                + "\n"
                + table._Footer()
            )

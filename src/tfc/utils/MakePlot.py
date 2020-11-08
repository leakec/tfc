import os
import pickle
import numpy as np
import matplotlib as matplotlib

# Change matplotlib backend to allow fig.show(). 
# Do not do this if READTHEDOCS is building.
if not os.environ.get('READTHEDOCS') == 'True':
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .TFCUtils import TFCPrint
TFCPrint()

class MakePlot():

    def __init__(self,xlabs,ylabs,twinYlabs=None,titles=None,zlabs=None,name='name'):
        # Set the fontsizes and family
        smallSize = 16
        mediumSize = 18
        largeSize = 18
        plt.rc('font', size=smallSize)
        plt.rc('axes', titlesize=mediumSize)
        plt.rc('axes', labelsize=largeSize)
        plt.rc('xtick', labelsize=mediumSize)
        plt.rc('ytick', labelsize=mediumSize)
        plt.rc('legend', fontsize=smallSize)
        plt.rc('figure', titlesize=largeSize)

        # Create figure and store basic labels
        self.fig = plt.figure()
        self._name = name

        # Consistify all label types
        if isinstance(xlabs,np.ndarray):
            pass
        elif isinstance(xlabs,str):
            xlabs = np.array([[xlabs]])
        elif isinstance(xlabs,tuple) or isinstance(xlabs,list):
            xlabs = np.array(xlabs)
        else:
            TFCPrint.Error("The xlabels provided are not of a valid type. Please provide valid xlabels")
        if len(xlabs.shape) == 1:
            xlabs = np.expand_dims(xlabs,1)

        if isinstance(ylabs,np.ndarray):
            pass
        elif isinstance(ylabs,str):
            ylabs = np.array([[ylabs]])
        elif isinstance(ylabs,tuple) or isinstance(ylabs,list):
            ylabs = np.array(ylabs)
        else:
            TFCPrint.Error("The ylabels provided are not of a valid type. Please provide valid ylabels")
        if len(ylabs.shape) == 1:
            ylabs = np.expand_dims(ylabs,1)

        if not zlabs is None:
            if isinstance(zlabs,np.ndarray):
                pass
            elif isinstance(zlabs,str):
                zlabs = np.array([[zlabs]])
            elif isinstance(zlabs,tuple) or isinstance(zlabs,list):
                zlabs = np.array(zlabs)
            else:
                TFCPrint.Error("The zlabels provided are not of a valid type. Please provide valid zlabels")
            if len(zlabs.shape) == 1:
                zlabs = np.expand_dims(zlabs,1)

        if titles is not None:
            if isinstance(titles,np.ndarray):
                pass
            elif isinstance(titles,str):
                titles = np.array([[titles]])
            elif isinstance(titles,tuple) or isinstance(titles,list):
                titles = np.array(titles)
            else:
                TFCPrint.Error("The titles provided are not of a valid type. Please provide valid titles.")
            if len(titles.shape) == 1:
                titles = np.expand_dims(titles,1)

        if twinYlabs is not None:
            if isinstance(twinYlabs,np.ndarray):
                pass
            elif isinstance(twinYlabs,str):
                twinYlabs = np.array([[twinYlabs]])
            elif isinstance(twinYlabs,tuple) or isinstance(twinYlabs,list):
                twinYlabs = np.array(twinYlabs)
            else:
                TFCPrint.Error("The twin ylabels provided are not of a valid type. Please provide valid twin ylabels")
            if len(twinYlabs.shape) == 1:
                twinYlabs = np.expand_dims(twinYlabs,1)


        # Create all subplots and add labels
        if zlabs is None:
            n = xlabs.shape
            self.ax = list()
            count = 0
            for j in range(n[0]):
                for k in range(n[1]):
                    if xlabs[j,k] is None:
                        continue
                    self.ax.append(self.fig.add_subplot(n[0],n[1],j*n[1]+k+1))
                    self.ax[count].set_xlabel(xlabs[j,k])
                    self.ax[count].set_ylabel(ylabs[j,k])
                    count += 1
        else:
            n = xlabs.shape
            self.ax = list()
            count = 0
            for j in range(n[0]):
                for k in range(n[1]):
                    if xlabs[j,k] is None:
                        continue
                    self.ax.append(self.fig.add_subplot(n[0],n[1],j*n[1]+k+1,projection='3d'))
                    self.ax[count].set_xlabel(xlabs[j,k])
                    self.ax[count].set_ylabel(ylabs[j,k])
                    self.ax[count].set_zlabel(zlabs[j,k])
                    count += 1
        
        if twinYlabs is not None:
            self.twinAx = list()
            count = 0
            for j in range(n[0]):
                for k in range(n[1]):
                    if xlabs[j,k] is None:
                        continue
                    self.twinAx.append(self.ax[count].twinx())
                    self.twinAx[count].set_ylabel(twinYlabs[j,k])
                    count += 1

        # Add titles if desired
        if titles is not None:
            count = 0
            for j in range(n[0]):
                for k in range(n[1]):
                    if titles[j,k] is None:
                        continue
                    self.ax[count].set_title(titles[j,k])
                    count += 1

        # Set tight layout for the figure
        self.fig.tight_layout()

    def FullScreen(self):

        # Get screensize
        import tkinter as tk
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        # Get dpi and set new figsize
        dpi = float(self.fig.get_dpi())
        self.fig.set_size_inches(width/dpi,height/dpi)

    def PartScreen(self,width,height):

        # Get screensize
        self.fig.set_size_inches(width,height)

    def show(self):
        self.fig.show()

    def save(self,fileName,transparent=True,fileType='pdf'):
        self.fig.savefig(fileName+'.'+fileType, bbox_inches='tight', pad_inches = 0, dpi = 300, format=fileType, transparent=transparent)

    def savePickle(self,fileName):
        pickle.dump(self.fig,open(fileName+'.pickle','wb'))

    def saveAll(self,fileName,transparent=True,fileType='pdf'):
        self.save(fileName,transparent=transparent,fileType=fileType)
        self.savePickle(fileName)

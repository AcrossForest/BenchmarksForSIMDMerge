from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fromClipboard():
    data = pd.read_clipboard()
    print(data)
    return data
    
    
def saveFigure(fig:Figure):
    f = plt.gcf()
    plt.savefig()
    

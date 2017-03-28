# -*- coding: utf-8 -*-

"""
1. We create a function which will plot the ROC curve
2. We create a function which will plot the bar chart for importance of variables
3. We create a function which creates a grid plot of two graphs
4. We create a function which creates a multiline ROC curve
"""

#import bokeh as bk
from bokeh.charts import *
import numpy as np
#from numpy import *
from bokeh.models import HoverTool
from bokeh.palettes import *
from bokeh.plotting import figure,output_file
from bokeh.charts.attributes import CatAttr
import pandas as pd


# Function plots the roc curve for a model. Pass the roc variable of the model
# as an argument, and returns the plot.
def show_roc(roc):
    my_array=np.array(roc)
    test=pd.DataFrame(my_array,index=['xaxis','yaxis']).T
    test['yaxis']=1-test['yaxis']
    tooltips=[('X','$x'),('Y','$y')]
    plot=Line(test,x='xaxis',y='yaxis', title="ROC Curve", tooltips=tooltips, xlabel='beta bogus (% of legit insulted)', ylabel='beta legit (% of bogus missed)', color=Spectral4[0])
    output_file('roc.html')
    return plot    
    #show(plot)

# Function plots the bar chart showing the importance of all the variable for a model
def show_varimp(varimp,title):
    d2=varimp
    plot2=Bar(d2,values='percentage',plot_width=800,plot_height=800, label=CatAttr(columns='variable', sort=False), title=title, ylabel="Percentage Explained", color=Spectral4[1])
    output_file('varimp.html')
    return plot2

# Function makes a grid which shows the roc curve and the importance of all the
# variables. We need to pass the trained model, and hte destination file names to 
# return the plot. 
#Arguments:
    # Model, path where graph should be saved, title of graph, number of variables whose importance 
    # should be seen

def analyze_model(model,of='Graphs/gridplot.html',title='Checking importance of variables',n_rows=None):
    plot=show_roc(model.roc(valid=1))
    data=model.varimp(use_pandas=True)
    data=data.sort_values(by='percentage',ascending=False)
    # Reduce the number of variables if entered by user, check the number entered is smaller
    # than maximum number of variables
    if n_rows!=None and n_rows<data.shape[0]:
        data=data.head(n=n_rows)
    plot2=show_varimp(data,title)    
    plot3=gridplot([[plot,plot2]])
#    plot3.title=title
    output_file(of)
    return plot3

def multiline_plot_addline(plot, x,y, color,legend):
    plot.line(x=x,y=y,color=color,line_width=3,legend=legend)
#def compare_models(models=None,model2=None,model3=None,model4=None,model5=None,model6=None,model7=None, legend1='Basic Model',legend2='Model 2',legend3='Model 3',legend4='Model 4',legend5='Model 5',legend6='Model 6',legend7='Model 7',of='Graphs/comparison_plot.html',title='Comparing different models'):
def compare_models(models,legends,of='Graphs/comparison_plot.html',title='Comparing different models'): 
    hover=HoverTool(tooltips=[("(x,y)","($x,$y)")])
    plot=figure(plot_width=800,plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','reset','resize'], x_axis_label='Legit Insulted', y_axis_label='Bogus missed')
    colors = Spectral10[:10]+Purples9[:3]+YlOrRd9[6:7]+YlOrRd9[2:3]
    for i in xrange(len(models)):
        my_array=np.array(models[i].roc(valid=1))
        multiline_plot_addline(plot,my_array[0],1-my_array[1],color=colors[i], legend=legends[i])
    plot.title=title
    output_file(of)
    return plot


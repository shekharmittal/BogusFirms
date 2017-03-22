# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:21:47 2017

@author: shekh
"""



features = ['VatRatio','LocalVatRatio','TurnoverGross','TotalReturnCount','RefundClaimed', 'ZeroTaxCredit'] 
added_features=['MoneyDeposited','MoneyGroup', 'AllCentral', 'AllLocal', 'ZeroTax', 'ZeroTurnover']
model_improvement_check(rf_v1,train,valid,features,added_features,more_features=None)





result = pd.merge(returns, profiles, how='left', on=['DealerTIN'])


result = pd.merge(returns, profiles, how='left', on=['DealerTIN'], indicator=True)

result = pd.merge(returns, profiles, how='left', on=['DealerTIN'], indicator='profile_merge')


















rf_v1.roc(valid=1)
roc=pd.DataFrame(rf_v1.roc)
roc=pd.DataFrame(data=rf_v1.roc())
roc=pd.DataFrame(rf_v1.roc(valid=1))
roc=pd.DataFrame(data=rf_v1.roc(valid=1)[:,:])
roc=pd.DataFrame(data=rf_v1.roc(valid=1))
df=rf_v1.roc(valid=1)



train, valid, test = divide_train_test(fr)

var_y = 'bogus_online'
features1= ['TurnoverGross','TurnoverLocal', 'MoneyDeposited']
rf_v1.train(features1, 'bogus_online', training_frame=train, validation_frame=valid )
roc_list.append(rf_v1.roc(valid=1) )

features2= ['VatRatio','LocalVatRatio', 'MoneyGroup', 'PositiveContribution', 'InterstateRatio','TotalReturnCount']
rf_v1.train(features1+features2, 'bogus_online', training_frame=train, validation_frame=valid )
roc_list.append(rf_v1.roc(valid=1) )

features3= ['AllCentral','AllLocal', 'ZeroTurnover', 'CreditRatio','ZeroTaxCredit']
rf_v1.train(features1+features2+features3, 'bogus_online', training_frame=train, validation_frame=valid )
roc_list.append(rf_v1.roc(valid=1) )

labels = ['features1','features1+features2','features1+features2+features3']

my_array=np.array(roc_list)

my_array.shape
test=pd.DataFrame(my_array)

x=test.T
x.shape





rf_v1.varimp_plot
rf_v1.varimp_plot()
rf_v1.varimp()
rf_v1.varimp(use_pandas=True)
tooltips=[('X','$xaxis'),('Y','$yaxis')]
plot=Line(x,x='xaxis',y='yaxis', tooltips=tooltips)
show(plot)
plot=Line(x,x='xaxis',y='yaxis', tooltips=tooltips)
show(plot)



gridplot=gridplot([plot,plot2])
plot2=Bar(d2,values='percentage', label=CatAttr(columns='index'),sort=True)
from bokeh.charts.attributes import CatAttr
plot2=Bar(d2,values='percentage', label=CatAttr(columns='index'),sort=True)
plot2=Bar(d2,values='percentage',label=CatAttr(columns='index',sort=True))
show(gridplot)
gridplot=gridplot([[plot,plot2]])
from bokeh.charts import gridplot
gridplot=gridplot([[plot,plot2]])
show(gridplot)


from bokeh.charts import gridplot
gridplot=gridplot([[plot,plot2]])
show(gridplot)
plot3=figure(plot_width=800, plot_height=800)
plot3=figure(plot_width=800, plot_height=800)
plot3.multi_line(roc_list[0],roc_list[1],roc_list[2],color=['red','blue','green'],line_width=4)
show(plot3)
plot3.multi_line(roc_list[0],color=['red','blue','green'],line_width=4)
show(plot3)
plot3.multi_line(roc_list[0],color=['red'],line_width=4)
show(plot3)
roc_list[0]
plot3.multi_line([roc_list[0]],color=['red'],line_width=4)
show(plot3)
rf_v1.roc(valid=1)
x=rf_v1.roc(valid=1)
x
x.describe()
x.shape()
x
x[0]
array=[x[0],x[1]]
array
plot3.multi_line(array,color=['red'],line_width=4)
show(plot3)
show(plot)



test=pd.DataFrame(my_array2,index=['xaxis2','yaxis2'])
test=test.T

test['xaxis2']=pd.Series(my_array2[0],index=test.index)
test['yaxis2']=pd.Series(my_array2[1],index=test.index)

test['xaxis3']=pd.Series(my_array3[0],index=test.index)
test['yaxis3']=pd.Series(my_array3[1],index=test.index)

test['yaxis2']=1-test['yaxis2']
test['yaxis3']=1-test['yaxis3']

hover=HoverTool(tooltips=[("(x,y)","($x,$y)")])


plot3=figure(plot_width=800, plot_height=800, tools=[hover])
#plot3.xaxis.visible=True
#x_range=Range1d(0,1)
plot3.multi_line(xs=[test['xaxis2'],test['xaxis3'],my_array[0]],ys=[test['yaxis2'],test['yaxis3'],1-my_array[1]],color=Spectral4,line_width=1)
show(plot3)


#%%
#%%
df_1['a_'] = pd.Series(a_list, index=df_1.index)


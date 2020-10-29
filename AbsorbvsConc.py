#Linear Regression
#a.k.a y=mx+b
# libraries
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plotter
from sklearn import linear_model
# Data
df=pd.DataFrame({'x': range(1,11), 'y': np.random.randn(10), 'y2': np.random.randn(10)+range(1,11)})
 
# multiple line plot
plt.plot( 'x', 'y', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.legend()


x=np.array([.005,.0075,.01,.015,.020])
X=x.reshape(-1,1)
print('Time (Seconds')
print(x)
y=np.array([.09,.149,.200,.260,.348])
print('Absorbance')
print(y)

X=x.reshape(-1,1)
print('Time (Seconds)')






lm=linear_model.LinearRegression()#setting our linear regression as a variable
model=lm.fit(X,y)#Actually perform the regression
predictions=lm.predict(X)

fig=plotter.figure(figsize=(10,5))
plot1=fig.add_subplot(111)
plot1.plot(x,y,'ro',x,predictions,'g--')#ro is our red dot and b-- is our blue line
csfont={'fontname':'Times New Roman'}#Font
plot1.set_title('Absorbance Vs Concentration $\mathregular{[I_2]}$',fontsize=22,**csfont)#Title
plot1.set_xlabel('$\mathregular{[I_2]}$ (mol/L)',fontsize=16,**csfont)
plot1.set_ylabel('Absorbance',fontsize=16,**csfont)
plot1.set_xlim()#Limits for axis
plot1.set_ylim()
plot1.set_y2lim()
plot1.grid(True)
plot1.minorticks_on()
print(lm.coef_)#The slope of the data regression line a.k.a 'm' from the formula
print(lm.intercept_)#This is the intercept of the line a.k.a 'b' from the formula
print(lm.score(x.reshape(-1,1),y))





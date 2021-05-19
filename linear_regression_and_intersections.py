# -*- coding: utf-8 -*-

# -- Sheet --


import pandas as pd
#import numpy as np 
import math as mt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression



lm = LinearRegression()
excel_file = 'parameters.xlsx'
df_excel = pd.read_excel(excel_file, sheet_name ="Arkusz 1", usecols='D')
df_excel_Cu = pd.read_excel(excel_file, sheet_name ="Arkusz 1", usecols='E')
k = int(1+3.3*(mt.log10(50)))
r = 1.9
szer_klasy = r/k

print(df_excel.describe())
print(df_excel.mean(), "średnia")
print(df_excel.median(), "mediana")
print(df_excel.std(), "odchylenie standardowe")
print(df_excel.mode(), "moda")
print(df_excel.var(), "wsp.wariancji")
print(df_excel.skew(), "wsp.skośności")

print("to jest to")
print(mt.tan(45-(46.79/2)))
    


reg_liniowa = sns.regplot(x =df_excel, y =df_excel_Cu, color='green' )

lm.fit(df_excel, df_excel_Cu )
print(lm.intercept_)
print(lm.coef_)
print(lm.score(df_excel, df_excel_Cu ))



#values = df_excel.hist("CuAg", facecolor="green", edgecolor="white", alpha = 0.75, bins = k )
#plt.xlabel("MIĄŻSZOŚĆ CuAg")
#plt.ylabel("CZĘSTOŚĆ")
#plt.ylabel("Cu")
#print(df_excel)
#print(szer_klasy)


#plt.savefig('CuAg')

plt.savefig('regplot1')


import pandas as pd
#import numpy as np 
import math as mt
import matplotlib.pyplot as plt
 


excel_file = 'parameters.xlsx'
plt.figure(figsize = (30, 30))
df_excel = pd.read_excel(excel_file, sheet_name ="Arkusz 1", usecols='E')

k = int(1+3.3*(mt.log10(50)))
r = (df_excel.max()- df_excel.min())
szer_klasy = r/k


values = df_excel.hist("Cu", facecolor="yellow", edgecolor="white", alpha = 0.75, bins = k )
plt.xlabel("ZAWARTOŚĆ Cu")
plt.ylabel("CZĘSTOŚĆ")
#plt.ylabel("Cu")
#print(df_excel)

print(df_excel.mean(), "średnia")
print(df_excel.median(), "mediana")
print(df_excel.std(), "odchylenie standardowe")
print(df_excel.mode(), "moda")
print(df_excel.var(), "wsp.wariancji")
print(df_excel.skew(), "wsp.skośności")


#print("DO HISTOGRAMU:")
#   print(r)
#print(df_excel.max())
#print(df_excel.min())
plt.savefig("CuHistogram")

import pandas as pd
import math as mt
import matplotlib.pyplot as plt


excel_file = 'parameters.xlsx'
plt.figure(figsize = (30, 30))
df_excel = pd.read_excel(excel_file, sheet_name ="Arkusz 1", usecols='F')

k = int(1+3.3*(mt.log10(50)))
r = (df_excel.max()- df_excel.min())
szer_klasy = r/k


values = df_excel.hist("Ag", facecolor="brown", edgecolor="white", alpha = 0.75, bins = k )
plt.xlabel(" ZAWARTOŚĆ Ag")
plt.ylabel("CZĘSTOŚĆ")

print(df_excel.mean(), "średnia")
print(df_excel.median(), "mediana")
print(df_excel.std(), "odchylenie standardowe")
print(df_excel.mode(), "moda")
print(df_excel.var(), "wsp.wariancji")
print(df_excel.skew(), "wsp.skośności")
#plt.ylabel("Cu")
#print(df_excel)
print(szer_klasy)
print(df_excel.max())
print(df_excel.min())
plt.savefig("AgHistogram")

y=0.508333

while y < 3.05:
  y=y+ 0.508333 
  print(y)

import pandas as pd
#import numpy as np 
import math as mt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression



lm = LinearRegression()
excel_file = 'parameters.xlsx'
#plt.figure(figsize = (30, 30))
df_excel = pd.read_excel(excel_file, sheet_name ="Arkusz 1", usecols='D')
df_excel_Ag = pd.read_excel(excel_file, sheet_name ="Arkusz 1", usecols='F')
k = int(1+3.3*(mt.log10(50)))
r = 1.9
szer_klasy = r/k

print(df_excel.describe())
print(df_excel.mean(), "średnia")
print(df_excel.median(), "mediana")
print(df_excel.std(), "odchylenie standardowe")
print(df_excel.mode(), "moda")
print(df_excel.var(), "wsp.wariancji")
print(df_excel.skew(), "wsp.skośności")


    


reg_liniowa = sns.regplot(x =df_excel, y =df_excel_Ag, color='brown' )

lm.fit(df_excel, df_excel_Ag )
print(lm.intercept_)
print(lm.coef_)
print(lm.score(df_excel, df_excel_Ag ))




plt.savefig('regplot2')

import pandas as pd
#import numpy as np 
import math as mt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression



lm = LinearRegression()
excel_file = 'parameters.xlsx'
#plt.figure(figsize = (30, 30))
df_excel = pd.read_excel(excel_file, sheet_name ="Arkusz 1", usecols='E')
df_excel_Ag = pd.read_excel(excel_file, sheet_name ="Arkusz 1", usecols='F')
k = int(1+3.3*(mt.log10(50)))
r = 1.9
szer_klasy = r/k

print(df_excel.describe())
print(df_excel.mean(), "średnia")
print(df_excel.median(), "mediana")
print(df_excel.std(), "odchylenie standardowe")
print(df_excel.mode(), "moda")
print(df_excel.var(), "wsp.wariancji")
print(df_excel.skew(), "wsp.skośności")




reg_liniowa = sns.regplot(x =df_excel, y =df_excel_Ag, color='magenta' )

lm.fit(df_excel, df_excel_Ag )
print(lm.intercept_)
print(lm.coef_)
print(lm.score(df_excel, df_excel_Ag ))




plt.savefig('regplot3')

import tensorflow as tf

import pandas as pd
import numpy as np 
from shapely.geometry import  LineString
import matplotlib.pyplot as plt

inter_y = [-1.0, 20.0]
inter_x = [1.7142857143, 1.7142857143]

excel_file = 'projekt1.xlsx'
df_aPrzezB= pd.read_excel(excel_file, sheet_name ="Arkusz1", usecols ='A')
df_a= pd.read_excel(excel_file, sheet_name ="Arkusz1", usecols ='B')
df_b= pd.read_excel(excel_file, sheet_name ="Arkusz1", usecols ='C')

plt.figure(figsize = (10, 7))
plt.plot(df_aPrzezB, df_b)
plt.plot(df_aPrzezB, df_a)
plt.plot(inter_x, inter_y, linestyle = ':', color= "black" )
#plt.axvline(1.7142857143)

line_1 = LineString(np.column_stack((df_aPrzezB, df_b)))
line_2 = LineString(np.column_stack((inter_x, inter_y)))
line_3 = LineString(np.column_stack((df_aPrzezB, df_a)))
intersection_1 = line_1.intersection(line_2)
intersection_2 = line_3.intersection(line_2)

plt.plot(*intersection_1.xy, 'o')
plt.plot(*intersection_2.xy, 'o')
plt.title("Wykres afa i beta")
plt.xlabel(" a/b")
plt.ylabel("alpha   beta")

plt.savefig("Bogusia")
plt.show()

print(intersection_1.xy)
print(intersection_2.xy)


import pandas as pd
import matplotlib.pyplot as plt

dolna_czesc = [20.909, 38.109]
gorna_czesc = [5.47, 5.47]


excel_file = 'boggy.xlsx'

df_promien1= pd.read_excel(excel_file, sheet_name ="Arkusz1", usecols ='A')
df_sigma_r1= pd.read_excel(excel_file, sheet_name ="Arkusz1", usecols ='B')
df_sigma_t1= pd.read_excel(excel_file, sheet_name ="Arkusz1", usecols ='C')

df_promien2= pd.read_excel(excel_file, sheet_name ="Arkusz1", usecols ='E')
df_sigma_r2= pd.read_excel(excel_file, sheet_name ="Arkusz1", usecols ='F')
df_sigma_t2= pd.read_excel(excel_file, sheet_name ="Arkusz1", usecols ='G')

plt.figure(figsize = (10, 7))
plt.plot(df_promien1, df_sigma_r1, color ="magenta")
plt.plot(df_promien1, df_sigma_t1, color = "black")
plt.plot(df_promien2, df_sigma_r2, color ="magenta")
plt.plot(df_promien2, df_sigma_t2, color = "black")
plt.plot(gorna_czesc, dolna_czesc, color= "black" )
#plt.axvline(1.7142857143)

#line_1 = LineString(np.column_stack((df_aPrzezB, df_b)))
#line_2 = LineString(np.column_stack((inter_x, inter_y)))
#line_3 = LineString(np.column_stack((df_aPrzezB, df_a)))
#intersection_1 = line_1.intersection(line_2)
#intersection_2 = line_3.intersection(line_2)

#plt.plot(*intersection_1.xy, 'o')
#plt.plot(*intersection_2.xy, 'o')
plt.title("WYKRES NAPRĘŻEŃ WTÓRNYCH W STREFACH OBLICZENIOWYCH WOKÓŁ WYROBISKACH")
plt.xlabel("PROMIEŃ [m]")
plt.ylabel("NAPRĘŻENIA [MPa]")
plt.legend()

plt.savefig("Bogusia")
plt.show()

#print(intersection_1.xy)
#print(intersection_2.xy)

q = pow((3/5), 3.215)
e = (2.4/3.215)
r = (4.39+e)*q-e
print(r)


# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:38:10 2022

@author: Naveen Kumar
"""
#########################1###################################
#import collections library used to find frequency
import collections
a = ['Naveen',1,2,1.2,1.3,1+2j,1==2]
b = ['Sandeep',3,4,3.3,3.4,3+4j,3==4]

c = [*a,*b]
d = a+b
d

# using Counter to find frequency of elements
frequency = collections.Counter(c)
dict(frequency)
c.reverse() # list reverse
######################2############################

A = {1,2,3,4,5,6,7,8,9,10}
B = {5,6,7,8,9,10,11,12,13,14,15}

C = A.intersection(B)
D = A^B #except common elements
A.remove(7)
B.remove(7)

############################3#################################

States = {'Andhra Pradesh':1000, 'Telangana':2000,'Maharastra':3000,'Karnataka':4000,'Punjab':5000}
print(States.keys())
States['Tamilnadu'] = 7000


###########################Module2##################################

x = 399
y = 543
z = 12345

equation = 22*y+x

if equation==z:
    print('It is a valid equation')
else:
    print('It is not a valid equation')
    
5//3 #Floor division
-5//3

#########################2######################
a = int(5)
b = int(3)
c = int(10)
a/=b #a = a/b
c*=5 #c= c*5

#####################3###################################

v = 'Data Science'
for i in v[:]:
    if i=='S':
        print('S is present')
    else:
        False
        
x = 4
y = 3

z = pow(x,y)
z
del(z)
######################Module4###############
Age = int(input('Enter your age:  '))
if Age<10:
    print('Children')
elif Age>60:
    print('Senior Citizen')
else:
    print('Normal Citizen')
    
Gender = str(input('Enter male/female: '))
Citizen = str(input('Enter senior/normal citizen: '))
Normal_Fare = float(input('Enter normal fare: '))
if str(Gender) == 'male' and str(Citizen) == 'senior':
    Fare = Normal_Fare*(0.7)
    print(Fare)
elif str(Gender) == 'female' and str(Citizen) == 'senior':
    Fare = Normal_Fare*(0.5)
    print(Fare)
elif str(Gender) == 'female' and str(citizen) == 'normal':
    Fare = Normal_Fare*(0.5)
    print(Fare)
else:
    print(Normal_Fare)
    

Number = float(input('Enter a number: '))
if Number%5==0:
    print('It is divided by 5')
else:
    print('Not divide by 5')

##################Module5####################

list1 = [1,5.5,(10+20j),'data science']
print(list1)
for i in range(len(list1)):
    print(type(list1[i]))
    
import numpy as np
np.arange(0,100,1)
n = int(input('Enter a number: '))
np.arange(0,n,1)

list1 = [0,1,2,3,4,5,6,7,8,9]
list2 = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
len(list1)
len(list2)
for i in range(len(list1)):
    list3 = dict({list1[i]:list2[i]})
    print(list3)

list1 = [3,4,5,6,7,8]
list2 = []
type(list2)
for i in range(5):
    if list1[i]%2==0:
        n = list1[i]+10
        list2.insert(i,n)
    elif list1[i]%2==1:
        m = list1[i]*5
        list2.insert(i,m)
print(list2)

message = input("Enter your name: ")
your_message = input('Enter your message: ')

if len(your_message) == 0:
    print("%s % s % s"%("Hello",message,'How are you?'))
else:
    print("%s % s % s"%("Hello",message,your_message))
    
    
###############Module6###########################

import pandas as pd
df = pd.read_csv(r"C:\Users\Naveen Kumar\Desktop\Data Science\Assignments\Python Problem Statements\Indian_cities.csv")
df.head()
df.info()

x = df.sex_ratio.nlargest(n=10)
y = x.index.tolist()
for i in y:
    print("sex ratio: ", df.sex_ratio[i]])
    print("City name: ", df.name_of_city[i])
    print("State code is :",df.state_code[i])
    print("State name is :", df.state_name[i])
    
a = df.total_graduates.nlargest(n=10)
b= a.index.tolist() 
for i in b:
    print("No.of graduates: ", df.total_graduates[i])
    print("City name: ", df.name_of_city[i])
    print("State code is :",df.state_code[i])

c = df.effective_literacy_rate_total.nlargest(n=10)
d = c.index.tolist()
for i in d:
    print("literacy rate: ", df.effective_literacy_rate_total[i])
    print("City name: ", df.name_of_city[i])
    print("location :",df.location[i])

import matplotlib.pyplot as plt
import seaborn as sns
df.info()
plt.hist(df.literates_total)
###########right skewed and leptokurtic
sns.scatterplot(data = df, x= df.male_graduates,y = df.female_graduates)
sns.boxplot(df.effective_literacy_rate_total)
df.isna().sum()

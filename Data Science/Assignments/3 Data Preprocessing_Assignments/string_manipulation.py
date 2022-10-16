# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:07:15 2022

@author: Naveen Kumar
"""
#######1#######
a = 'Grow Gratitude'
a[0]
len(a)
a.count('G')
#######2#######
b = 'Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else'
len(b)
len(b.split())
####3####
c = 'Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth'
c[0]
c[0:3]
c[-3:]
####4####
d = 'stay positive and optimistic'
e = d.split()
for i in range(len(e)):
    if e[i].startswith('H'):
        print('Yes')
    else:
        print('No')

for i in range(len(e)):
    if e[i].endswith('d'):
        print('Yes')
    else:
        print('No')

for i in range(len(e)):
    if e[i].endswith('c'):
        print('Yes')
    else:
        print('No')

######5#######
f ='🪐'

for i in range(108):
    print(f)

#######7####
g = 'Grow Gratitude'
h = g.split()
h[0] = h[0].replace('Grow','Growth')
g = h[0] + h[1]
g
print("% s % s"%(h[0],h[1]))

######8########
j = '.elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A'
j[::-1]

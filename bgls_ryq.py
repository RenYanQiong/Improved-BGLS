# -*- coding: utf-8 -*-
#================================================================================
# Copyright (c) 2014 João Faria, Annelies Mortier
# Distributed under the MIT License.
# (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
#================================================================================
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import numpy as np
import sys

import matplotlib.pyplot as plt
import math
import time
import pylab as pl

# from numba.cgutils import printf
from astropy.units import count
from pylab import *  
from matplotlib.ticker import MultipleLocator, FormatStrFormatter 
from numpy import average

# from scipy.optimize import newton
# from sklearn.utils.testing import assert_almost_equal
try:
	import mpmath  # https://code.google.com/p/mpmath/
except ImportError, e1:
	try:
		from sympy import mpmath  # http://sympy.org/en/index.html
	except ImportError, e2:
		raise e2
	finally:
		raise e1

pi = np.pi
starttime = time.clock()


class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False
    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

def bgls(t, y, err, plow, phigh, ofac):
	f = np.linspace(1./phigh, 1./plow, int(100*ofac))
	#f = np.linspace(0.25, 50, 19900)

	omegas = 2. * pi * f

	err2 = err * err
	w = 1./err2
	W = sum(w)

	bigY = sum(w*y)  # Eq. (10)

	p = []
	constants = []
	exponents = []

	for i, omega in enumerate(omegas):
		theta = 0.5 * np.arctan2(sum(w*np.sin(2.*omega*t)), sum(w*np.cos(2.*omega*t)))
		x = omega*t - theta
		cosx = np.cos(x)
		sinx = np.sin(x)
		wcosx = w*cosx
		wsinx = w*sinx

		C = sum(wcosx)
		S = sum(wsinx)

		YCh = sum(y*wcosx)
		YSh = sum(y*wsinx)
		CCh = sum(wcosx*cosx)
		SSh = sum(wsinx*sinx)

		if (CCh != 0 and SSh != 0):
			K = (C*C*SSh + S*S*CCh - W*CCh*SSh)/(2.*CCh*SSh)

			L = (bigY*CCh*SSh - C*YCh*SSh - S*YSh*CCh)/(CCh*SSh)

			M = (YCh*YCh*SSh + YSh*YSh*CCh)/(2.*CCh*SSh)

			constants.append(1./np.sqrt(CCh*SSh*abs(K)))

		elif (CCh == 0):
			K = (S*S - W*SSh)/(2.*SSh)

			L = (bigY*SSh - S*YSh)/(SSh)

			M = (YSh*YSh)/(2.*SSh)

			constants.append(1./np.sqrt(SSh*abs(K)))

		elif (SSh == 0):
			K = (C*C - W*CCh)/(2.*CCh)

			L = (bigY*CCh - C*YCh)/(CCh)

			M = (YCh*YCh)/(2.*CCh)

			constants.append(1./np.sqrt(CCh*abs(K)))

		if K > 0:
			raise RuntimeError('K is positive. This should not happen.')

		exponents.append(M - L*L/(4.*K))

	constants = np.array(constants)
	exponents = np.array(exponents)

	logp = np.log10(constants) + (exponents * np.log10(np.exp(1.)))

	p = [10**mpmath.mpf(x) for x in logp]

	p = np.array(p) / max(p)  # normalize

	p[p < (sys.float_info.min * 10)] = 0
	p = np.array([float(pp) for pp in p])

	return 1./f, p

#------------------------------Improved BGLS by RENYQ-----------------------


def searchPrd(t,y,err): 	#search rough results by BGLS,time,mag,errors
	TimeSpace=[0.001,1,2,3,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20]
# 	TimeSpace=[0.1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	Periods = []
	for i in range(0,len(TimeSpace)-1):		
		f, p = bgls(t, y, err,TimeSpace[i],TimeSpace[i+1],10)
		ind = 0		
		for i in p:
			if i==1:
				Periods.append(f[ind]) # for comets or other stars
		    	ind += 1 
	return Periods		

def foldedLc(t,P):#-----------------input(time data,period)
	temp=t%P
	cof = np.polyfit(temp,y,12) 
	p=np.poly1d(cof)  
	y1=p(temp)
	return temp,y1,p

def ShowFoldedLC(Periods, t,y,chi_2 ):# show folded lightcurves for different periods
	fig11 = plt.figure(11)
	fig11.canvas.set_window_title('0-5 hours')
 	
	chi2_temp=sorted(chi_2)
	for i in range(0,len(Periods)/2):
		ax = fig11.add_subplot(2, 3, i+1)
		plt.plot(t%Periods[i],y,'ob')
		plt.xlim(0, np.max(t%Periods[i])) 
		xmajorLocator   = MultipleLocator(0.4*(i+1))
		ax.xaxis.set_major_locator(xmajorLocator)  
	plt.title(' Folded Lightcurves from 0-5 hours',position=(0.5, 2.3),size=18)	
	
	fig12 = plt.figure(12)
	fig12.canvas.set_window_title('5-10 hours')
	j = 0
	for i in range(len(Periods)/2,len(Periods)):
		j=j+1
		ax = fig12.add_subplot(2, 3, j)
		plt.plot(t%Periods[i],y,'ob')
		plt.xlim(0, np.max(t%Periods[i])) 
		xmajorLocator   = MultipleLocator(0.2*(i+1))
		ax.xaxis.set_major_locator(xmajorLocator)  
	plt.title(' Folded Lightcurves from 5-10 hours',position=(0.5, 2.3),size=18)




#---------------------------Newton's method iteration
def newton(dp,ddp,x):#( coefficients,formulation of derived function)
	dcof=np.polyder(dp) # coefficients of Second Derived function
	ddcof= np.poly1d(dcof)# formulation of Second Derived function
 	x0=x #    x is initial value
	while abs(ddp(x0))>0.001:
		x0=x0-ddp(x0)/ddcof(x0)
	temp=x0
	return temp


def fits(t,y,err):
	Periods=searchPrd(t,y,err)
	Periods=np.array(Periods)*2# For asteroids
# 	Periods=np.array(Periods) # For comets or other objects
	chi_2=[]#sum of (y-yi)^2
	temp_x=0

	
#-----------------------------------obtain chi_2 and find minimum 
	for P in Periods:
		temp,y1,Funs=foldedLc(t,P)
		chi_2.append(sum((y-y1)**2))
	chi_2=np.array(chi_2)

	
	ShowFoldedLC(Periods/2, t,y,chi_2 )
#-------------------------------------sort chi_2 and Periods from small to large
	for i in range(0,len(chi_2)):
		for j in range(0,i):
			if chi_2[i]<=chi_2[j]:
				chi_2[i],chi_2[j]=chi_2[j],chi_2[i]
				Periods[i],Periods[j]=Periods[j],Periods[i]
	T_min1 = Periods[0]
	T_min2 = Periods[1]

	
	
#------------------------------------obtain predict period by newton	

	T_temp = []
	T = []
	temp = t%Periods[0]# folded points
	cof = np.polyfit(temp,y,12) #Polynomial coefficients
	p=np.poly1d(cof)  #formulation of Polynomial function
	dp = np.polyder(cof)#coefficients of derived function
	ddp = np.poly1d(dp) # formulation of derived function
	x=np.linspace(0.001,Periods[0],10)
	for x_new in x:
		prd_newton=newton(dp,ddp,x_new)
		if prd_newton  and prd_newton<=Periods[0]:
			T_temp.append(prd_newton)
	T_temp=np.array(T_temp)
	
	testtemp=p(T_temp)

	

	
#----------------------------------------sort values of f(x)=0 from small to large
	T_temp.sort()
	Temporary=T_temp[0]
	T.append(T_temp[0])
	ind = 0
	for i in range(1,len(T_temp)):
		#------------
# 		if abs(p(Temporary)-p(T_temp[i]))>(max(T_temp)-min(T_temp))/10 and T_temp[i] not in T:
# 		if abs(p(Temporary)-p(T_temp[i]))>(mean(T_temp))/10 and T_temp[i] not in T:
		if abs(p(Temporary)-p(T_temp[i]))>0.01 and T_temp[i] not in T:
			T.append(T_temp[i])
			Temporary=T_temp[i]
			ind += 1
		else:
			if p(Temporary)<p(T_temp[i]) :
				T[ind]=T_temp[i]
	T_pre = (T[len(T)-1]-T[0])/(len(T)-1)*4
	
	print 'Predict Period is: %f hours' % (T_pre) ,T
	



	Chi_2=[]
	PreP =np.linspace(T_pre-1,T_pre+1,4000)# refine results
	for j in range(4000):
		temp,y1,Funs=foldedLc(t,PreP[j])
		Chi_2.append(sum((y-y1)**2))# merit function
	position=Chi_2.index(min(Chi_2))
	print 'Best Period is: %f hours' % (PreP[position]) 
	
	P_best = PreP[position]

	
	
#----------------------------plot the relationship between chi^2 and periods
	fig2 = plt.figure(2)	
	plt.plot(PreP,Chi_2,'.r')
	plt.title('Distribution of' + r' '+'$ \mathrm{x}^{2}$'+'',size=20)
	plt.xlabel('Periods / Hours',size = 20)
	plt.ylabel(''+r' '+'$ \mathrm{x}^{2}$'+'', size = 20)

	
	TEMP,y2,p=foldedLc(t,PreP[position] )

	fig3 = plt.figure (3)
	plt.plot(TEMP,y,'*b',TEMP,y2,'or')
	plt.title('Folded LC Fitting',size=20)
	plt.xlabel('Time / Hours',size = 20)
	plt.ylabel('Brightness', size = 20)

	return P_best
#------------------------------Example-----------------------------------

def ReadPMOdata():
#	filenames = ReadFilename()
#	for FileName in filenames:
	FileName = '128314.txt'
#	FileName = ReadFilename()
	fid = open( FileName, 'r')
	JDT = []
	MAG = []
	ERR = []
	for line in fid:
		fields = line.split()
# 		JDT.append(float(fields[0]))
# 		MAG.append(float(fields[3]))
# 		ERR.append(float(fields[2]))
		JDT.append(float(fields[0]))
		MAG.append(float(fields[1]))
		ERR.append(float(fields[2]))
	fid.close()
	return JDT, MAG, ERR

JDT,MAG,ERR = ReadPMOdata()
t=(np.array(JDT)-min(JDT))*24
# y = np.array(MAG)
y = [1000000/(10**((x-1)/(2.5))) for x in MAG]
err = np.array(ERR)


plt.figure(num=1, figsize=(8,6))
plt.title(' Light Curves', size = 20)
plt.xlabel('Time(hours)', size = 20)
plt.ylabel('Brightness', size = 20)
plt.plot(t, y, color = 'b', linestyle =
	'--', marker='o',label='y1 data')


P_best = fits(t,y,err) 	
  
endtime = time.clock()	
print 'use time: %f seconds' % (endtime-starttime) 
plt.show()



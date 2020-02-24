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
import os

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
	# phigh = 4
	# plow = 3
	f = np.linspace(1./phigh, 1./plow, int(100*ofac))

	# f = np.arange(1./phigh, 1./plow, 0.001)

	# f = np.arange(1. / phigh, 1. / plow, 0.001)

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
	TimeSpace=[0.01,1,2,3,4,5,6,7,8,9,10]
	# TimeSpace = np.linspace(0, 40, 41)
	# TimeSpace[0] = 0.01

	Periods = []
	for i in range(0,len(TimeSpace)-1):
		f, p = bgls(t, y, err, TimeSpace[i], TimeSpace[i+1], 10)
		ind = 0
		for i in p:
			if i == 1:
				Periods.append(f[ind])
		    	ind += 1
	return Periods

def foldedLc(t,P):#-----------------input(time data,period)
	temp = t % P
	cof = np.polyfit(temp,y,12) # Polynomial coefficients
	p=np.poly1d(cof)  # equation
	y1=p(temp)
	return temp, y1, p

def ShowFoldedLC(Periods, t, y, chi_2): # show folded lightcurves for different periods
	fig11 = plt.figure(11)
	fig11.canvas.set_window_title('0-5 hours')

	chi2_temp=sorted(chi_2)
	for i in range(0,len(Periods)/2):
		ax = fig11.add_subplot(2, 3, i+1)
		plt.plot(t%Periods[i],y,'ob')
		plt.xlim(0, np.max(t%Periods[i]))
		xmajorLocator   = MultipleLocator(0.4*(i+1))
		ax.xaxis.set_major_locator(xmajorLocator)
	plt.title(' Folded Lightcurves from 0-5 hours', position=(0.5, 2.3),size=18)

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


def fits(t,y,err,AsterNum):
	Periods=searchPrd(t,y,err)
	Periods=np.array(Periods)
	chi_2=[]#sum of (y-yi)^2
	temp_x=0


#-----------------------------------obtain chi_2 and find minimum
	for P in Periods:
		temp,y1,Funs=foldedLc(t,P*2)# fold up use P*2
# 		temp,y1,Funs=foldedLc(t,P) #fold up use P
		chi_2.append(sum((y-y1)**2))
	chi_2=np.array(chi_2)
	minindex  = np.argmin(chi_2)
# ------------------------------ period*2
	p_temp = Periods[minindex]*2 #for asteroid
# 	p_temp = Periods[minindex]   #for other objects, e.g. comet
#-------------------------------------------------------------------
#
# 	ShowFoldedLC(Periods, t,y,chi_2 )
	P_best = p_temp
	TEMP,y2,p=foldedLc(t,p_temp )


##
	Chi_2 = []
	PreP = np.linspace(0.01, 10, 10000)  # refine results
	for j in range(10000):
		temp, y1, Funs = foldedLc(t, PreP[j]*2)
		Chi_2.append(sum((y - y1) ** 2)/10000)  # merit function
	position = Chi_2.index(min(Chi_2))
	# 	print 'Best Period is: %f hours' % (PreP[position])

	# P_best = PreP[position]


	# fig2 = plt.figure(2)
	# plt.plot(PreP,Chi_2,'.r')
	# plt.title('Distribution of' + r' '+'$ \mathrm{x}^{2}$'+'',size=20)
	# plt.xlabel('Periods / Hours',size = 20)
	# plt.ylabel(''+r' '+'$ \mathrm{x}^{2}$'+'', size = 20)

	fig3 = plt.figure (3)
	points = np.linspace(0, P_best, len(TEMP)*1.5)
	plt.plot(TEMP, y, '*b', TEMP, p(TEMP), 'or')
	plt.title('Period of Asteroid ' + str(AsterNum) +' = '+ '%.2f'%P_best+' hr', size=20)#str(P_best)
	plt.xlabel('Time / Hours',size = 20)
	plt.ylabel('Magnitude', size = 20)
	#plt.savefig('steps0.001/'+ str(AsterNum) + '_Folded LC Fitting.png')
	# plt.show()
	plt.close()
	# plt.show()

	return P_best, chi_2[minindex]
#------------------------------Example-----------------------------------

def ReadPMOdata(AsterNum):
	FileName = str(AsterNum)+".txt"
	fid = open( FileName, 'r')
	JDT = []
	MAG = []
	ERR = []
	HelioDis = []
	GeoDis = []
	PhaseAngle = []
	AbsMag = []
	for line in fid:
		fields = line.split()
		JDT.append(float(fields[0]))#Julian Date from MJD
		# MAG.append(float(fields[1]))#apparent magnitude
		MAG.append(float(fields[3])) #Magnitude corrected for distance and phase function
		ERR.append(float(fields[2]))#uncertainty in apparent magnitude
		# HelioDis.append(float(fields[10]))#Heliocentric distance
		# GeoDis.append(float(fields[11]))#Geocentric distance
		HelioDis.append(float(fields[5]))#Heliocentric distance
		GeoDis.append(float(fields[7]))#Geocentric distance
		PhaseAngle.append(float(fields[8]))  # solar phase angle

	fid.close()
	return JDT, MAG, ERR, HelioDis, GeoDis, PhaseAngle



##   read asteroid ID from txt file

f = open('U_2.txt','r')
ind = 0
# y = []
for line in f:
	fields = line.split()
	AsterNum = int(fields[0])
	if os.path.isfile(str(AsterNum)+'.txt'):
		starttime = time.clock()
		JDT, MAG, ERR, HelioDis, GeoDis, PhaseAngle = ReadPMOdata(AsterNum)
		Index_outliner = []
		MAG_temp = MAG-np.min(MAG)
		VarMAG = np.var(MAG_temp)
		AveMAG = mean(MAG_temp)
		Reduce_mag = []
		Phi = np.array(PhaseAngle)
		# Phi = np.array(PhaseAngle)
		# Phi1 = [np.exp((-3.33)*(np.tan(0.5*x))**0.63) for x in Phi]
		# Phi2 = [np.exp((-1.87)*(np.tan(0.5*x))**1.22) for x in Phi]
		for i in range(0, len(JDT)):
			# Reduce_mag.append(MAG[i]+5*math.log(HelioDis[i]*GeoDis[i])+2.5*math.log((1-0.15)*Phi1[i]+0.15*Phi2[i]))  #reduced magnitude
			Reduce_mag.append(MAG[i]) #Magnitude corrected for distance and phase function
			if (MAG_temp[i]) > AveMAG*3 or (AveMAG/3) > MAG_temp[i]:
				Index_outliner.append(i)
# 		print len(Index_outliner),len(Reduce_mag)
		T_t = (np.array(JDT)-min(JDT))*24
		Y_y = np.array(Reduce_mag)
		# print len(T_t)
		# y = [1000000/(10**((x-1)/(2.5))) for x in Reduce_mag]
		Err_err = np.array(ERR)
		t = np.delete(T_t, Index_outliner)
		y = np.delete(Y_y, Index_outliner)
		err = np.delete(Err_err, Index_outliner)
# 		print len(t)


		#
		# plt.figure(num=1, figsize=(8,6))
		# plt.title(' Light Curves', size = 20)
		# plt.xlabel('Time(hours)', size = 20)
		# plt.ylabel('Brightness', size = 20)
		# plt.plot(t, y, color = 'b', linestyle =
		# 	'--', marker='o',label='y1 data')


		P_best, loss = fits(t,y,err,AsterNum)
		print ('%.2f' %P_best, '%.6f'%loss)

		endtime = time.clock()
		print 'use time: %f seconds' % (endtime-starttime)


# 		print 'The period of Asteroid', AsterNum, 'is: ',  P_best
	else:
# 		print 'No information of Asteroid', AsterNum
		print 0
	plt.show()
	ind += 1
f.close()


# AsterNum = 5823
# # JDT,MAG,ERR = ReadPMOdata(AsterNum)
# JDT, MAG, ERR, HelioDis, GeoDis, PhaseAngle = ReadPMOdata(AsterNum)
# Index_outliner = []
# MAG_temp = MAG-np.min(MAG)
# VarMAG = np.var(MAG_temp)
# AveMAG = mean(MAG_temp)
# Reduce_mag = []
# Phi = np.array(PhaseAngle)
# # print HelioDis
# # print GeoDis
# # print  tan(0.5*Phi)
# # Phi1 = exp((-3.33)*(tan(0.5*Phi))**0.63)
# # Phi2 = [np.exp(-1.87*(np.tan(0.5*x))**1.22) for x in Phi]
#
# for i in range(0, len(JDT)):
# 	# Reduce_mag.append(MAG[i] - 5 * np.log(HelioDis[i] * GeoDis[i]))
# 	# Reduce_mag.append(MAG[i]-5*np.log(HelioDis[i]*GeoDis[i])+2.5*np.log((1-0.15)*Phi1[i]+0.15*Phi2[i]))  #absolute magnitude
# 	Reduce_mag.append(MAG[i]) #Magnitude corrected for distance and phase function
# 	if (MAG_temp[i]) > AveMAG*3 or (AveMAG/3) > MAG_temp[i]:
# 		Index_outliner.append(i)
# # print len(Index_outliner),len(Reduce_mag), len(JDT),Index_outliner
# print Reduce_mag
# T_t = (np.array(JDT)-min(JDT))*24
# Y_y = np.array(Reduce_mag)
# # print len(T_t)
# # y = [1000000/(10**((x-1)/(2.5))) for x in Reduce_mag]
# Err_err = np.array(ERR)
# t = np.delete(T_t, Index_outliner)
# y = np.delete(Y_y, Index_outliner)
# err = np.delete(Err_err, Index_outliner)
# # print len(t)
#
# # plt.figure(num=1, figsize=(8,6))
# # plt.title(' Light Curves', size = 20)
# # plt.xlabel('Time(hours)', size = 20)
# # plt.ylabel('Brightness', size = 20)
# # plt.plot(t, y, color = 'b', linestyle =
# # 	'--', marker='o',label='y1 data')
# P_best = fits(t, y, err)
# print 'The period of Asteroid', AsterNum, 'is: ',  P_best
# plt.show()

# 
#endtime = time.clock()
#print 'use time: %f seconds' % (endtime-starttime)

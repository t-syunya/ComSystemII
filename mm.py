#!/usr/bin/env python3

import sys
import os
import subprocess
import time

#array size set
n = 960	

#matrics intialize
A = [[0 for i in range(n)] for j in range(n)]
B = [[0 for i in range(n)] for j in range(n)]
C = [[0 for i in range(n)] for j in range(n)]

for i in range(n):
	for j in range(n):
		# Python's float type is 64 bit double-precision floating point data
		A[i][j] = B[i][j] = float(i+j)/(j+0.5) 

# print( type(A[0][0]))

#gemm calculation
start = time.time() 
for i in range(n):
	for j in range(n):
		for k in range(n):
			C[i][j] += A[i][k] * B[k][j] 
elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) , "[sec]")

# convert second to hour, minute and seconds
elapsed_time = int(elapsed_time)
elapsed_hour = elapsed_time // 3600
elapsed_minute = (elapsed_time % 3600) // 60
elapsed_second = (elapsed_time % 3600 % 60)

# print as 00h00m00s
print(str(elapsed_hour).zfill(2) , "h " , str(elapsed_minute).zfill(2) , "m " , str(elapsed_second).zfill(2), "s " )



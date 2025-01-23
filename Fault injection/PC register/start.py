#!/usr/bin/python
# -*- coding: UTF-8 -*-

import gdb
import multiprocessing
import time
import psutil

import re
import random
import sys
import os
import signal

round_value = 0
process_pid = 0

class TimeoutProcessKiller(gdb.Command):

	def __init__(self):
		 # 4. Register the name of the command in the constructor
		 super(self.__class__, self).__init__("killer", gdb.COMMAND_USER)

	def getPIDFromName(self, processName):
		file = open("k.out", 'w', encoding='utf-8')
		pids = psutil.pids()
		for pid in pids:
			try: # Use try blocks to prevent discovered processes from being revoked and resulting in psutil NoSuchProcess error causing process revocation
				p = psutil.Process(pid)
				# file.write("name1: %s\n" % p.name())
				# file.write("name2: %s\n" % processName.split('/')[-1])
				if p.name() == processName.split('/')[-1]:
					return pid
			except psutil.NoSuchProcess:
				continue
		file.close()
		return 0

	def Killer(self, times = 3,  processName = "CDBVA_rad2deg"):
	# def Killer(self, times = 3, processName = "qsort_smallcfsig"):
		'''
		'''
		file = open("kill.out", 'w', encoding='utf-8')
		file.write("start: %s|\n" % processName)
		count = 0
		while 1:
			pid_old = self.getPIDFromName(processName)
			# # global process_pid
			# pid_old = process_pid
			file.write("pid: %d\n" % pid_old)
			try:
				time.sleep(int(times))
			except KeyboardInterrupt:
				exit()
			pid_new = self.getPIDFromName(processName)
			# pid_new = process_pid
			if pid_old == pid_new and pid_old != 0:
				count += 1
				try:
					process_kill = psutil.Process(pid_old)
				except psutil.NoSuchProcess:
					continue
				process_kill.terminate()
				os.kill(pid_new, signal.SIGKILL)
				# process_kill.join()
				file.write("kill %d\n" % pid_new)
		file.close()


	# def invoke(self, args, from_tty):
	#     argv = gdb.string_to_argv(args)
	#     # self.Killer(argv[0], argv[1])

class Start(gdb.Command):
	'''
		start time programe cycles 
	'''

	def __init__(self):
		super(self.__class__, self).__init__("start",gdb.COMMAND_USER)

	def invoke(self, args, from_tty):
		argv = gdb.string_to_argv(args)
		if len(argv)<4 or len(argv)>5:
			raise gdb.GdbError("start parameter error")
		tp = TimeoutProcessKiller()
		p_tp = multiprocessing.Process(target = tp.Killer, args = (argv[0], argv[1]))
		p_tp.start()
		
		# print("argv[2]:", argv[2])
		for i in range(int(argv[2])):
			global round_value
			round_value = i
			# print("round_value", round_value)
			fi = Fault_inject()
			p_fi = multiprocessing.Process(target = fi.invoke, args = ("%c %s" % (argv[3], 1), from_tty))
			p_fi.start()
			global process_pid
			process_pid = p_fi.pid
			p_fi.join()
		try:
			p_tp.terminate()
		except KeyboardInterrupt:
			exit()





#hex2bin
def hex2dec(string_num):
	return int(string_num,16)

#bin2hex
def dec2hex(num):
	return hex(num)
#####################################################


corr_file = open('inject.correct','r')
corr = corr_file.read().replace('\n','')

##############################################################
class Fault_inject(gdb.Command):
	"""docstring for ClassName"""
	
	def __init__(self):
		# 4. Register the name of the command in the constructor
		
		super(self.__class__, self).__init__("inject", gdb.COMMAND_USER)

	def bitsize(self,bits):
		realhex = re.search(r'(?<=0x)(\d|\w)+',bits).group()
		binlen = 4*len(realhex)
		return binlen

	def conversion(self,address,bitsize,flip_bits):
		for _ in range(flip_bits):
			randbit = random.randint(0,bitsize-1)
			convert = (2**randbit)^(hex2dec(address))
		return dec2hex(convert)



	def fault_inject(self,*injectpoint):
		if len(injectpoint) == 3:
			pc_convert = self.conversion(injectpoint[0],injectpoint[1],injectpoint[2])
			print(pc_convert)
			gdb.execute('set $pc='+pc_convert)
		elif len(injectpoint) == 4:
			randomadd = random.randint(0,len(injectpoint[1])-1)
			gdb.execute('set $pc='+ injectpoint[1][randomadd])
			print(injectpoint[1][randomadd])
		


	def invoke(self, args, from_tty):
		argv = gdb.string_to_argv(args)

		find_error = 0
		seg_fault = 0
		corr_ans = 0

		if len(argv)!=2:
			raise gdb.GdbError("parameter error，\n \
				input 'help inject' to view details")

#################################get pc info#########################
		pclist = []
		pccodelist = []
		pattern = re.compile(r'0x(\d|\w)+')
		mainpoint = gdb.execute('b main',to_string=True)
		gdb.execute('run',to_string=True)
		pcs = gdb.execute('disassemble',to_string=True)

		pcs = pcs.split('\n')
		for i in range(len(pcs)):
			pcv = pattern.search(pcs[i])
			if pcv:
				pclist.append(pcv.group())              
				pccodelist.append(pcs[i])
		gdb.execute('delete',to_string=True)
			

######################fault inject##############################
		# print("argv[1]",argv[1])
		for i in range(int(argv[1])):
			print('\n')
			global round_value
			print('round %d:'% (round_value+1))
			randbreak = random.randint(0,len(pclist)-1) 
			# print('inject at instruction: \n'+pccodelist[randbreak])
			# print(randbreak)
			injectpoint = gdb.execute('b *'+pclist[randbreak],to_string=True)
			# injectpoint = gdb.execute('b *'+pclist[randbreak])
			
			print('#')
			brp = gdb.execute('run')
			# brp = gdb.execute('run')
			# print('$$$'+brp+'$$$')
			# print('&'+brp+'&')
			try:
				# pcinfo = gdb.execute('i r pc',to_string=True)
				pcinfo = gdb.execute('i r pc',to_string=True)

			except:
				pcinfo=''
			finally:
				if pcinfo:
					pcadd = pcinfo.split()[1]
					b_size = self.bitsize(pclist[randbreak])   #compute bitsize 64 or 32
					if argv[0] == 'p':
						if len(argv)==5:
							self.fault_inject(pcadd,pclist,b_size,argv[4])
						else:
							self.fault_inject(pcadd,pclist,b_size,1)
					elif argv[0] == 'r':
						if len(argv)==5:
							self.fault_inject(pcadd,b_size,argv[4])
						else:
							self.fault_inject(pcadd,b_size,1)
					else:
						raise gdb.GdbError("parameter error，\n \
							input 'help inject' to view details")
					gdb.execute('delete')

					out = gdb.execute('continue',to_string=True).replace('\n','')
					#############################################
					all_ans = out + str(brp)
					print(all_ans)
					#############################################
					if re.findall(r'(Segmentation.*fault)',out):
						seg_fault+=1
					elif re.findall(r'(control.*error)',out):
						find_error+=1
					# elif        
				print('#')
				print('end round')
		print(seg_fault)
		print(find_error)





Start() 

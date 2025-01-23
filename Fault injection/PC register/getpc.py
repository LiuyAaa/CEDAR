import gdb
import numpy as np

class Get_PC_Value(gdb.Command):
	"""
	Obtain the PC value of the fault injection point
	"""
	def __init__(self):
		super(self.__class__, self).__init__("getpcvalue", gdb.COMMAND_USER)

	def invoke(self, args, from_tty):
		argv = gdb.string_to_argv(args)
		if len(argv)!=1:
			raise gdb.GdbError("parameter error，\n \
				input 'help inject' to view details")

		f1 = open('pc_value.txt', 'w', encoding='utf-8')
		gdb.execute('set pagination off')
		# gdb.execute('b main',to_string=True)
		gdb.execute('watch gSig')
		gdb.execute('r '+argv[0])
		round =0
		bbnumlist = []
		pc = gdb.execute('p $pc',to_string = True).split(' ')[-2]
		blocknum = gdb.execute('p gSig',to_string = True).split('\n')[0].split(' ')[-1]
		print('pc:',pc)
		print('bnum:',blocknum)
		table  = []
		while blocknum != -1:
			table.append([int(blocknum),pc])
			bbnumlist.append(int(blocknum))
			gdb.execute('ni')
			try:
				pc = gdb.execute('p $pc',to_string = True).split(' ')[-2]
			except:
				break
			blocknum = gdb.execute('p gSig',to_string = True).split('\n')[0].split(' ')[-1]
			print('round',round)
			print('pc:',pc)
			print('bnum:',blocknum)
			round = round+1
		gdb.execute('info b')
		gdb.execute('d')
		#Deduplication of arrays
		s = set() #Create an empty collection
		for t in table:
			s.add(tuple(t)) 
		list2 = np.array(list(s))
		list3 = sorted(list2,key = lambda x:int(x[0]))
		for i in range(len(list3)):
			try:
				f1.write(str(list3[i][0])+':'+str(list3[i][1])+"\n")
			except:
				continue


		bbnumset = list(set(bbnumlist))
		bbnumset.sort()
		bbnumcount = []
		# print(bbnumset)
		f2 = open('block_runtimes.txt', 'w', encoding='utf-8')
		for i in range(len(bbnumset)):
			bbnumcount.append(bbnumlist.count(bbnumset[i]))
			f2.write(str(bbnumset[i])+":"+str(bbnumcount[i])+"\n")
		f2.close()
		f1.close()
		print('end')

Get_PC_Value()

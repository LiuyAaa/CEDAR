#!/usr/bin/python
# -*- coding: UTF-8 -*-

import gdb
import re
import random
import sys
import time
import multiprocessing
import time
import psutil

class TimeoutProcessKiller(gdb.Command):

    def __init__(self):
         # 4.          # 4. Register the name of the command in the constructor

         super(self.__class__, self).__init__("killer", gdb.COMMAND_USER)

    def getPIDFromName(self, processName):
        pids = psutil.pids()
        for pid in pids:
            try: # Use try blocks to prevent discovered processes from being revoked and resulting in psutil NoSuchProcess error causing process revocation
                p = psutil.Process(pid)
            except psutil.NoSuchProcess:
                continue
            if p.name() == processName:
                return pid
        return 0

    def Killer(self, times = 3, processName = "qsort_smallcfsig"):
        '''
        '''
        count = 0
        while 1:
            pid_old = self.getPIDFromName(processName)
            time.sleep(int(times))
            pid_new = self.getPIDFromName(processName)
            if pid_old == pid_new and pid_old != 0:
                count += 1
                try:
                    process_kill = psutil.Process(pid_old)
                except psutil.NoSuchProcess:
                    continue
                process_kill.terminate()
                print("kill %d"%count)

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
        if len(argv) != 4:
            raise gdb.GdbError("start parameter error")
        tp = TimeoutProcessKiller()
        p_tp = multiprocessing.Process(target = tp.Killer, args = (argv[0], argv[1]))
        # tp.Killer(argv[0], argv[1])
        fi = Fault_inject()
        p_fi = multiprocessing.Process(target = fi.invoke, args = ("%c %s" % (argv[3], argv[2]), from_tty))
        p_fi.start()
        p_tp.start()
        p_fi.join()
        try:
            p_tp.terminate()
        except psutil.NoSuchProcess:
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
	for bits in range(flip_bits):
            randbit = random.randint(0,bitsize-1)
            convert = (2**randbit)^(hex2dec(address))
        return dec2hex(convert)



    def fault_inject(self,*injectpoint):
        if len(injectpoint) == 3: #r
            pc_convert = self.conversion(injectpoint[0],injectpoint[1],injectpoint[2])
            gdb.execute('set $pc='+pc_convert)
        elif len(injectpoint) == 4: #p
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
        for i in range(int(argv[1])):
            print('\n')
            print('round %d:'%(i+1))
            randbreak = random.randint(0,len(pclist)-1) 
            print('inject at instruction: \n'+pccodelist[randbreak])
            # print(randbreak)
            # injectpoint = gdb.execute('b *'+pclist[randbreak],to_string=True)
            injectpoint = gdb.execute('b *'+pclist[randbreak])
            
            print('#')
            # brp = gdb.execute('run',to_string=True)
            brp = gdb.execute('run')
            # print('$$$'+brp+'$$$')
            # print('&'+brp+'&')
            try:
                pcinfo = gdb.execute('i r pc',to_string=True)

            except:
                pcinfo=''
            finally:
                if pcinfo:
                    pcadd = pcinfo.split()[1]
                    b_size = self.bitsize(pclist[randbreak])   #compute bitsize 64 or 32
                    if argv[0] == 'p':
                        if argv[2]:
                            self.fault_inject(pcadd,pclist,b_size,argv[2])
                        else:
                            self.fault_inject(pcadd,pclist,b_size,1)
                    elif argv[0] == 'r':
                        if argv[2]:
                            self.fault_inject(pcadd,b_size,argv[2])
                        else:
                            self.fault_inject(pcadd,b_size,1)	
                    else:
                        raise gdb.GdbError("parameter error，\n \
                            input 'help inject' to view details")
                    gdb.execute('delete')

                    out = gdb.execute('continue',to_string=True).replace('\n','')
                    #############################################
                   all_ans = out + brp
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
                            # print(pcs)
Start() 

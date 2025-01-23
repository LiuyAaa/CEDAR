import gdb
import random

base = [str(x) for x in range(10)] + [ chr(x) for x in range(ord('A'),ord('A')+6)]
# hex2dec
def hex2dec(string_num):
    return str(int(string_num.upper(), 16))
 
# dec2bin
def dec2bin(string_num):
    num = int(string_num)
    mid = []
    while True:
        if num == 0: 
            break
        num,rem = divmod(num,2)
        mid.append(base[rem])
    return ''.join([str(x) for x in mid[::-1]])

# hex2tobin
def hex2bin(string_num):
    return dec2bin(hex2dec(string_num.upper()))

class Fault_inject(gdb.Command):
 
    """Fault_inject
    Usage: inject old_breakpoint_num new_breakpoint
    Example:
        (gdb) mv 1 binary_search -- move `breakpoint` 1 to `b binary_search`
    """
    
    def __init__(self):
        # 4. Register the name of the command in the constructor
        super(self.__class__, self).__init__("inject", gdb.COMMAND_USER)
 
    # 5. Implement the specific functions of this custom command in the invoke method
    # Args represents the parameters that are concatenated after the command, where string is used_ To_ Argv converted to an array

    def coversion(self,address):
        position = random.randint(0,len(address))
        print(position)
        if address[position] == '0':
            address = address[:position]+'1'+address[position+1:]
        else:
            address = address[:position]+'0'+address[position+1:]
        return address

    def fault_inject(self,*location):
        if len(location) == 2:
            com = 'b'+' '+location[0]+':'+location[1]
            gdb.execute(com,to_string = True)
            gdb.execute('run',to_string=True)

        else:
            com = 'b'+' '+location[0]
            bp = gdb.execute(com,to_string=True)
            run_info = gdb.execute('run',to_string=True)
            pc = gdb.execute('i r pc',to_string=True)
            pcadd = pc.split()[1]
            print(pcadd)
            bina_pc = hex2bin(pcadd)
            # print(bina_pc)
            pc_convert = self.coversion(bina_pc)
            # print(pc_convert)
            print(pc)
            


    def invoke(self, args, from_tty):
        argv = gdb.string_to_argv(args)
        if argv[0] != '-f' and argv[0] != '-l':
            raise gdb.GdbError("parameter error，\n \
                input 'help inject' to view details")
        if argv[0] == '-l':
            if len(argv)>3:
                raise gdb.GdbError("Too many parameter, \
                    input 'help inject' to view details")
            try:
                argv[1]
                argv[2]
            except:
                print('sigle file fault injecting at range('+argv[1]+')')
                self.fault_inject(argv[1])
            else:
                self.fault_inject(argv[1],argv[2])
        if argv[0] == '-f':
        	pass

Fault_inject()      
 
# 7. Register the custom command with the gdb session

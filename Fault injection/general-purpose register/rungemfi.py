#!/usr/bin/python
import os
import subprocess
import time
import sys

work_dir = os.getcwd()
cur_dir = ""


program_name = sys.argv[2]


def cmd(cmdstr):
    
    p = subprocess.Popen(cmdstr, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cur_dir, shell=True)
    p.wait()
    return p.stdout.read()


def run(cmdstr, timeout):

    p = subprocess.Popen(cmdstr, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cur_dir, shell=True)
    T = 1

    while T < timeout:

        time.sleep(1)
        if(p.poll() is not None):
            return p.stdout.read()
        T = T + 1
    if(p.poll() is None):
        return "timeout"


def writefile(file, content):
    f = open(cur_dir+"/"+file, mode='w')
    f.write(content)
    f.close()


gem_cmd = "/home/yy2/res/gemFI/build/ARM/gem5.opt  /home/yy2/res/gemFI/configs/example/se.py -c " + program_name
fiini = work_dir+"/dofi/fi.ini"

# print(work_dir)
# init
def runFI(count):
    global cur_dir
    cur_dir = work_dir
    cmd("rm -r fi")
    cmd("mkdir fi")

    cur_dir = work_dir+"/fi"
    # run golden
    cmd("mkdir golden")
    cur_dir = work_dir+"/fi/golden"

    cmd("cp "+work_dir+"/nofi/fi.ini .")

    writefile("output.txt", run(gem_cmd,  10))



    # do fi
    for i in range(1, count+1):
        print "do fi " , str(i)
        cur_dd = "fi_" + str(i)
        cur_dir = work_dir+"/fi"
        cmd("mkdir " + cur_dd)
        cur_dir = work_dir+"/fi/"+cur_dd
	print cur_dir
        cmd("cp " + fiini + " .")

        writefile("output.txt", run(gem_cmd, 10))

    
if __name__ == "__main__" :
    print work_dir
    runFI(int(sys.argv[1]))




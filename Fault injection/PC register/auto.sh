#!/bin/bash
# 自动注入故障并统计故障的个数
# 用法: $0 $1待注入程序 $2程序的输入文件 $3运行次数 $4p/r故障注入方式 $5守护进程检测时间间隔 $6统计脚本缩写(d,i,...)
cp $1 /home/yy2/Desktop/qt/CPUControl/fault_inject
cp $2 /home/yy2/Desktop/qt/CPUControl/fault_inject
cd /home/yy2/Desktop/qt/CPUControl/fault_inject

#echo "get pc value"
#echo "file $1" > getpc.input
#echo "so getpc.py" >> getpc.input
#echo "getpcvalue "${2##*/} >> getpc.input
#gdb < getpc.input >/dev/null
#echo "get pc value done"

echo "file $1" > inject.input
echo "so start.py" >> inject.input
echo "set args ${2##*/}" >> inject.input
echo "start $5 $1 $3 $4 $6" >> inject.input
echo "gdb"
gdb < inject.input > inject.output
echo "correct"
./$1  ${2##*/} > inject.correct
# 把输入输出文件的编码格式改为 utf-8 否则python3 的 readline() 报错
 echo "convert corret output file"
vim inject.correct +"set fileencoding=UTF-8" +wq!
# 把输入输出文件的编码格式改为 utf-8 否则python3 的 readline() 报错
echo "convert output file"
vim inject.output +"set fileencoding=UTF-8" +wq!


#!/bin/bash 
# Preprocessing source program, redundant fragile instructions, instrumentation control flow detection instructions

#Variable declaration
fileName=$1
fileNameIR=${fileName%.*}.ll
fileNameWithId=${fileName%.*}_id.ll
fileNameWithDmr=${fileName%.*}_dmr.ll
fileNameWithGlobalSig=${fileName%.*}_gs.ll
fileNameWithSplit=${fileName%.*}_split.ll
fileNameWithCfc=${fileName%.*}_dmr_cf.ll
assemblyLanguageSourceProgram=${fileName%.*}_dmr_cf.s
executableProgram=${fileName%.*}_dmr_cf

#Generate IR file
clang $fileName -emit-llvm -g -S -o  $fileNameIR

#Insert LLFIindex
opt -load /home/yy2/Desktop/qt/CCQ/datasource/cross_compile/mem/libgenLLFIIndex.so -genllfiindexpass -S $fileNameIR -o $fileNameWithId

#Hardening instructions
opt -load /home/yy2/Desktop/qt/CCQ/datasource/cross_compile/mem/libdmr.so -cmp -select_set_file=index_ins.txt -S $fileNameWithId -o $fileNameWithDmr


#Add global signatures
opt -load /home/yy2/Desktop/llvm/build/lib/AddGlobalSig.so -AddGlobalSig -S $fileNameWithDmr -o $fileNameWithGlobalSig

#Generate index of BBs which need to be split
/home/yy2/Desktop/llvm/build/bin/opt -load /home/yy2/Desktop/llvm/build/lib/Index.so -Index -disable-output $fileNameWithGlobalSig 2> allindex.txt
python3 sort.py

#Split BB
for line in `cat index_bb.txt`
do
	/home/yy2/Desktop/llvm/build/bin/opt -load /home/yy2/Desktop/llvm/build/lib/SplitBlock.so -SplitBlock -S -index $line $fileNameWithGlobalSig -o  $fileNameWithSplit
done
echo 'split done'

#Signature insertion
/home/yy2/Desktop/llvm/build/bin/opt -load /home/yy2/Desktop/llvm/build/lib/SIG1.so -SIG1 -S $fileNameWithSplit -o $fileNameWithCfc

#Compile error handling functions into. o file
gcc exec.c -c -o exec.o

#Compile .ll file into .o file
llc $fileNameWithCfc -o $assemblyLanguageSourceProgram


#Generate executable program
gcc $assemblyLanguageSourceProgram exec.o -o $executableProgram

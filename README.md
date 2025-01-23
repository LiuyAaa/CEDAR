# CEDAR 

A novel approach for silent CFE detection based on heterogeneous relation learning, which detects silent CFEs by localizing and hardening vulnerable instructions.

# Runtime Environment：

 Ubuntu 16.04 with 4.15 kernel；
 
 LLVM 7.1;
 
 Pytorch 1.10.2.

# Usage:

(1) Execute the scripts in the "Vulnerable instruction localization" folder to train on node classification task:

```
python3 main.py
```

(2) Execute the scripts in the "Silent CFE detection" folder to harden programs:

```
./harden.sh ./program_name
```

(3) Execute the scripts in the "Fault injection" folder to inject faults:

```
./auto.sh ./program_name; Program parameters; Number of runs; Injection mode; Timeout;
```

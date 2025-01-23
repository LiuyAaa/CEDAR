# coding=UTF-8

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
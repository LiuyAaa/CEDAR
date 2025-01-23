#!/bin/bash

#auto set profiles, then run gemfi and statistic

#fcount=
#fitype=
#section=
#fun=
#The above four parameters need to be set by youself



#Check whether the relative address has ben provided or not
if [ $# -lt 1 ]; then
    echo "Usage: $0 <relative_path_to_program> <fi_times>"
    exit 1
fi

#Get absolute path
absolute_path=$(readlink -f "$1")
echo "Absolute path to the program: $absolute_path"

#Check whether the fi.ini exist or not
config_file="./nofi/fi.ini"
if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file $config_file not found."
    exit 1
fi

#Setting ./nofi/fi.ini
sed -i "s|^program=.*$|program=$absolute_path|" "$config_file"

echo "Program address in $config_file updated to: $absolute_path"

#Check whether the fi.ini exist or not
config_file1="./dofi/fi.ini"
if [ ! -f "$config_file1" ]; then
    echo "Error: Configuration file $config_file1 not found."
    exit 1
fi

#Setting ./dofi/fi.ini
sed -i "s|^program=.*$|program=$absolute_path|" "$config_file1"

echo "Program address in $config_file1 updated to: $absolute_path"

	#Setting profile location
profile_location=${absolute_path%/*}"/fi/golden/profile"
echo $profile_location

sed -i "s|^profile=.*$|profile=$profile_location|" "$config_file1"

echo "Program address in $config_file1 updated to: $profile_location"


#Run gemfi
python rungemfi.py $2 $absolute_path

#statistic
python statistic.py

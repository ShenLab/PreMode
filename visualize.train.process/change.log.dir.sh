#!/bin/bash
# $1 is the file path of the script
# $2: new log dir
cd /share/terra/Users/gz2294/RESCVE.final
logdir=$(cat $1 | grep log_dir | sed 's/.*: //')
sed -i "s|log_dir: "$logdir"|log_dir: "$2"/|g" $1
mv $logdir $2
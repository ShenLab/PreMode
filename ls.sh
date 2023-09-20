#!/bin/bash
logdir=$(cat $1 | grep log_dir | sed 's/.*: //')
echo $logdir
ls $logdir $2

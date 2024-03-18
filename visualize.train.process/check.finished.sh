#!/bin/bash
# $1 is the name of the scripts folder
# check if task has finished
logdir=$(cat $1 | grep log_dir | sed 's/.*: //')
num_epochs=$(cat $1 | grep num_epochs | sed 's/.*: //')
if [ -f $logdir/model.epoch.$num_epochs.pt ]; then
  echo "Finished "$1
  else
  echo "Not Finished "$1
fi

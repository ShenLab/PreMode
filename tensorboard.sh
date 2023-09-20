#!/bin/bash
logdir=$(cat $1 | grep log_dir | sed 's/.*: //')
tensorboard --logdir $logdir/log/ --port 8891

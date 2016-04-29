#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo $0" <conf_fn> [gpu=1]"
    exit
fi
gid=1
if [ $# -ge 2 ]; then
    gid=$2
fi
#THEANO_FLAGS="optimizer=fast_compile,mode=NanGuardMode,device=gpu$gid" python script_main_att.py $1
#THEANO_FLAGS="mode=NanGuardMode,device=gpu$gid" python script_main_att.py $1
THEANO_FLAGS="device=gpu$gid" python script_main_att.py $1

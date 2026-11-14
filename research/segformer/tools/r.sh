#!/bin/bash
# reuse one ssh master connection to the GPU server
O="-o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=15 -o ControlMaster=auto -o ControlPath=/tmp/rx_sc_%C -o ControlPersist=2h -o ServerAliveInterval=30 -i $HOME/.ssh/eai_seetacloud -p 13870"
H=root@connect.cqa1.seetacloud.com
if [ "$1" = "put" ]; then for i in 1 2 3 4 5; do scp -P 13870 ${O//-p 13870/} "$2" $H:"$3" && exit 0; sleep 3; done; exit 1; fi
for i in 1 2 3 4 5; do ssh $O $H "$@"; rc=$?; [ $rc -ne 255 ] && exit $rc; sleep 3; done; exit 255

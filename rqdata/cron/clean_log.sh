#!/bin/bash

LOG_FILE="/home/idc2/notebook/rqdata/Log/cron.log"
if [ -f "$LOG_FILE" ]; then
    /bin/rm -f "$LOG_FILE"
fi
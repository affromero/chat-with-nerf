#!/bin/sh

# $1 is process name
# $2 is memory limit in MB

# ./killIf.sh chrome 4000 (This will kill the chrome process if it exceeds 4GB)

if [ $# -ne 2 ];
then
    echo "Invalid number of arguments"
    exit 0
fi

while true;
do
    pgrep "$1" | while read -r procId;
    do
        SIZE=$(pmap $procId | grep total | grep -o "[0-9]*")
        SIZE=${SIZE%%K*}
        SIZEMB=$((SIZE/1024))
        SIZEGB=$((SIZEMB/1024))
        SIZE_95=$((($2*95)/100))
        # echo "Process id = $procId Size = $SIZEMB MB"
        if [ $SIZEMB -gt $2 ]; then
            kill -9 "$procId"
            echo "Killed the $1 process id = $procId using more than $SIZEGB GB RAM"
            exit 0
        else
            if [ $SIZEMB -gt $SIZE_95 ]; then
                echo "RAM SIZE is greater than 95% of the limit! Process id = $procId Size = $SIZEMB MB. Careful! Limit is $2 MB."
                echo "If you want to increase the limit, please run the command again with the new limit."
                echo "If you do not know where this limit is coming from, please contact the system administrator or check the ~/.bashrc file."
            fi
        fi
    done

    sleep 1
done
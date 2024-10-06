#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "Usage: ./measure_time.sh <command>"
    exit 1
fi

start_time=$(date +%s%N) # 记录命令执行前的纳秒级时间戳
eval "$@"
end_time=$(date +%s%N) # 记录命令执行后的纳秒级时间戳

elapsed_time=$((end_time - start_time)) # 计算时间差
elapsed_seconds=$(bc <<< "scale=9; $elapsed_time/1000000000") # 将纳秒转换为秒

echo "Execution time: $elapsed_seconds seconds"


#!/bin/bash
GIT_HASH=$(git log --pretty=format:'%h' -n 1)
echo $GIT_HASH

PREFIX="$PWD/opt/cachelib/"
LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH

CONFIG_FILES="config_kvcache config_kvcache_FIFO config_kvcache_ML config_kvcache_ML_FIFO memc_replay memc_replay_FIFO memc_replay_ML memc_replay_ML_FIFO"

for CONFIG_FILE in $CONFIG_FILES
do
taskset -c 0-31 ./build-cachelib/cachebench/cachebench \
--json_test_config ./cachelib/cachebench/test_configs/"$CONFIG_FILE".json \
>> ./cachelib/cachebench/test_configs/fb/"$CONFIG_FILE"."$GIT_HASH"
tail -n 10 ./cachelib/cachebench/test_configs/fb/"$CONFIG_FILE"."$GIT_HASH"
done
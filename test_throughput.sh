#!/bin/bash
if [ -z "$1" ]
then
  GIT_HASH=$(git log --pretty=format:'%h' -n 1)
else
  GIT_HASH="$1"
fi
echo $GIT_HASH

git stash
git checkout "$GIT_HASH"
./contrib/build-cachelib.sh
LOG_PATH="/nfs/$GIT_HASH/"
mkdir "$LOG_PATH"

PREFIX="$PWD/opt/cachelib/"
LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH

CONFIG_FILES="config_kvcache config_kvcache_LRU config_kvcache_FIFO config_kvcache_ML config_kvcache_ML_FIFO memc_replay memc_replay_LRU memc_replay_FIFO memc_replay_ML memc_replay_ML_FIFO"

for CONFIG_FILE in $CONFIG_FILES
do
if [ -e "./cachelib/cachebench/test_configs/"$CONFIG_FILE".json" ]
then
echo "./cachelib/cachebench/test_configs/"$CONFIG_FILE".json"
taskset -c 0-31 ./build-cachelib/cachebench/cachebench \
--json_test_config ./cachelib/cachebench/test_configs/"$CONFIG_FILE".json \
>> "$LOG_PATH$CONFIG_FILE"
tail -n 10 "$LOG_PATH$CONFIG_FILE"
fi
done

git checkout main
git stash pop
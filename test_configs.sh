#!/bin/bash
GIT_HASHES="90dc148ac5e1db995102ece577fc16cb9605c3b0 4b73ff594581bda0a9391fb7bb3b366d73228019"
for GIT_HASH in $GIT_HASHES
do
git checkout "$GIT_HASH"
./contrib/build-cachelib.sh
LOG_PATH="/proj/lrbplus-PG0/workspaces/yangdsh/$GIT_HASH/"
mkdir "$LOG_PATH"

PREFIX="$PWD/opt/cachelib/"
LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH

CONFIG_FILES="memc_replay memc_replay_ram"

for CONFIG_FILE in $CONFIG_FILES
do
if [ -e "/proj/lrbplus-PG0/workspaces/yangdsh/"$CONFIG_FILE".json" ]
then
taskset -c 0-31 ./build-cachelib/cachebench/cachebench \
--json_test_config /proj/lrbplus-PG0/workspaces/yangdsh/"$CONFIG_FILE".json \
>> "_$LOG_PATH$CONFIG_FILE"
tail -n 10 "_$LOG_PATH$CONFIG_FILE"
fi
done
done

git checkout main
R=/tmp/imgchk-$$; mkdir -p $R/{data,run,tmp}
export ENROOT_CACHE_PATH=$R/cache ENROOT_DATA_PATH=$R/data ENROOT_RUNTIME_PATH=$R/run ENROOT_TEMP_PATH=$R/tmp
enroot list; enroot create --name chk <WORK>/vime/enroot/vllm-cb12c30.sqsh >/dev/null 2>&1
enroot start --root --rw --rc <WORK>/wt/h_vi/h100-handoff/vime-dsv4-20261126/rc.sh chk bash -c 'ls -ld /root /root/FlashMLA; echo HOME=$HOME; python3 -m site; grep -E " /root| <FS>" /proc/mounts; ls <HOME>/.local/lib/python3.12/site-packages 2>&1 | head -20; ls /root/FlashMLA | head -3'
rm -rf $R

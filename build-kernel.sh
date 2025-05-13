#!/bin/bash 
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

branch='k1-bl-v2.2.y'

if [ ! -d "${SCRIPT_DIR}/toolchain" ]; then
    echo "no toolchain, please run env-prepare first"
    exit 1
fi

export PATH=${SCRIPT_DIR}/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.0.5/bin:$PATH
export CROSS_COMPILE=riscv64-unknown-linux-gnu-
export ARCH=riscv

build(){
    if [ ! -d "linux-6.6" ]; then
        git clone -b ${branch} https://gitee.com/bianbu-linux/linux-6.6.git
    else
        cd linux-6.6
        make k1_defconfig
        LOCALVERSION="" make -j$(nproc)
    fi
}

clean(){
    if [ -d "linux-6.6" ]; then
        cd linux-6.6
        make clean -j$(nproc)
    fi
}

# clean
build


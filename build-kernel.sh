#!/bin/bash 
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

username="litchi"
board_ip="192.168.31.233"
branch='k1-bl-v2.2.y'

if [ ! -d "${SCRIPT_DIR}/toolchain" ]; then
    echo "no toolchain, please run env-prepare first"
    exit 1
fi

export PATH=${SCRIPT_DIR}/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.0.5/bin:$PATH
export CROSS_COMPILE=riscv64-unknown-linux-gnu-
export ARCH=riscv

build_module(){
    cd ${SCRIPT_DIR}/linux-6.6
    make modules_install INSTALL_MOD_PATH=../output/
}

build_deb(){
    cd ${SCRIPT_DIR}/linux-6.6
    make -j$(nproc) bindeb-pkg
    mv ${SCRIPT_DIR}/*deb ${SCRIPT_DIR}/output
    mv ${SCRIPT_DIR}/*buildinfo* ${SCRIPT_DIR}/*changes* ${SCRIPT_DIR}/output

}

build(){
    if [ ! -d "${SCRIPT_DIR}/linux-6.6" ]; then
        git clone -b ${branch} https://gitee.com/bianbu-linux/linux-6.6.git
    else
        cd ${SCRIPT_DIR}/linux-6.6
        make k1_defconfig
        LOCALVERSION="" make -j$(nproc)
        build_module
        build_deb
    fi
}

clean(){
    if [ -d "${SCRIPT_DIR}/linux-6.6" ]; then
        cd ${SCRIPT_DIR}/linux-6.6
        make clean -j$(nproc)
    fi
}

scpToBoard(){
    scp -r ${SCRIPT_DIR}/output/* ${username}@${board_ip}:/home/${username}/
}

# clean
build
build_module
build_deb


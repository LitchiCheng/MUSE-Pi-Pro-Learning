#!/bin/bash 
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit
fi

echo "正在检查Ubuntu系统版本..."
get_version() {
    if command -v lsb_release &>/dev/null; then
        lsb_release -r | awk '{print $2}'
    elif [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$VERSION_ID"
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        echo "$DISTRIB_RELEASE"
    else
        echo "无法确定Ubuntu版本" >&2
        exit 1
    fi
}

VERSION=$(get_version)
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "错误：无法解析版本号: $VERSION"
    exit 1
fi

echo "当前Ubuntu版本: $VERSION"

MAJOR=$(echo "$VERSION" | cut -d. -f1)
MINOR=$(echo "$VERSION" | cut -d. -f2)

installPackages() {
    sudo apt-get update
    sudo apt-get install git build-essential cpio unzip rsync file bc wget python3 python-is-python3 libncurses5-dev libssl-dev dosfstools mtools u-boot-tools flex bison python3-pip -y
    sudo pip3 install pyyaml
    sudo apt install repo -y
    # 用于打包deb
    sudo apt install debhelper -y 
    sudo apt install asciidoc -y
}

if [ "$MAJOR" -gt "20" ]; then
    installPackages
elif [ "$MAJOR" -eq "20" ] && [ "$MINOR" -ge "04" ]; then
    installPackages
else
    sudo apt-get install git build-essential cpio unzip rsync file bc wget python3 libncurses5-dev libssl-dev dosfstools mtools u-boot-tools flex bison python3-pip -y
    sudo pip3 install pyyaml
    sudo apt install repo -y
fi

repoSync() {
    if [ ! -d "bianbu-linux-2.2" ]; then
        curl -L https://mirrors.tuna.tsinghua.edu.cn/git/git-repo -o repo
        chmod +x repo
        export REPO_URL='https://mirrors.tuna.tsinghua.edu.cn/git/git-repo'

        mkdir bianbu-linux-2.2
        cd bianbu-linux-2.2
        sudo ${SCRIPT_DIR}/repo init -u git@gitee.com:bianbu-linux/manifests.git -b main -m k1-bl-v2.2.y.xml
        sudo ${SCRIPT_DIR}/repo sync
        sudo ${SCRIPT_DIR}/repo start k1-bl-v2.2.y --all
    fi
}

downloadToolchain(){
    if [ ! -d "toolchain" ]; then
        if [ ! -f "toolchain.tar.xz" ]; then wget https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.0.5.tar.xz -O toolchain.tar.xz; fi
        mkdir toolchain
        sudo tar -Jxf toolchain.tar.xz -C toolchain
    fi
}

repoSync
downloadToolchain






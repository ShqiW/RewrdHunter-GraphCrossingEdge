#!/bin/bash
set -exu
if [ $# -gt 0 ]; then
    PROJECT_NAME=$1
else
    echo "Please provide a project name as the first argument." >&2
    echo "Usage: $0 <project_name>" >&2
    exit 1
    PROJECT_NAME=$(echo $HOME | awk -F'/' '{print $3}')
fi

cd $(mktemp -d)

echo "Setting .bashenv for cluster-specific bashprofile loading..."
if [ -f $HOME/.bashenv ]; then
    echo ".bashenv already exists."
else
    echo "
#!/bin/bash
export LOAD_BASHENV=true

# depending on the first character of host name, p or o, add the corresponding path
hostname=\$(hostname)

case \"\${hostname:0:1}\" in
p)
    source $HOME/.bashprofiles/pitzer.sh
    ;;
a)
    source $HOME/.bashprofiles/ascend.sh
    ;;
c)
    source $HOME/.bashprofiles/cardinal.sh
    ;;
*) ;;
esac
    " >$HOME/.bashenv
    sed -i '/source \$HOME\/.bashenv/d' $HOME/.bashrc
    echo "source \$HOME/.bashenv" >>$HOME/.bashrc
    echo ".bashenv set."
fi
source $HOME/.bashenv



RANDOM_SUFFIX=$(openssl rand -hex 4)

USERNAME=$(whoami)
if [ -d "$HOME/.bashprofiles/$CLUSTER_NAME" ]; then
    rm -rfv "$HOME/.bashprofiles"/$CLUSTER_NAME*
fi
mkdir -p "$HOME/.bashprofiles"
wget https://webdav.coredumped.tech/s/XFbjizQeLwntN2Q/download/bashprofiles.zip -O "$HOME/.bashprofiles/bashprofiles.zip"
unzip -o "$HOME/.bashprofiles/bashprofiles.zip" -d "$HOME/.bashprofiles"
mv "$HOME/.bashprofiles/bashprofiles"/$CLUSTER_NAME* "$HOME/.bashprofiles/"
rm -rfv $HOME/.bashprofiles/bashprofiles*
for file in "$HOME/.bashprofiles/"$CLUSTER_NAME*; do
    sed -i "s/PAS2330/$PROJECT_NAME/g" "$file"
    sed -i "s/__RANDOM_SUFFIX_TO_BE_REPLACED_BY_INITIALIZE_POETRY_SH__/${RANDOM_SUFFIX}/g" "$file"
done

if [ -d "/fs/scratch/$PROJECT_NAME/${USERNAME}/.config" ]; then
    rm -rf "/fs/scratch/$PROJECT_NAME/${USERNAME}/.config"/$CLUSTER_NAME*
fi
rm -rf $HOME/.config/pypoetry
mkdir -p "/fs/scratch/$PROJECT_NAME/${USERNAME}/.config"
wget https://webdav.coredumped.tech/s/eM8fc7Xt5nqsgeM/download/poetry_config.zip -O "/fs/scratch/$PROJECT_NAME/${USERNAME}/.config/poetry_config.zip"
unzip -o "/fs/scratch/$PROJECT_NAME/${USERNAME}/.config/poetry_config.zip" -d "/fs/scratch/$PROJECT_NAME/${USERNAME}/.config"
mv "/fs/scratch/$PROJECT_NAME/${USERNAME}/.config/poetry_config/${CLUSTER_NAME}_pypoetry" "/fs/scratch/$PROJECT_NAME/${USERNAME}/.config/${CLUSTER_NAME}_pypoetry"
rm -rf "/fs/scratch/$PROJECT_NAME/${USERNAME}/.config"/poetry_config*

for file in /fs/scratch/$PROJECT_NAME/${USERNAME}/.config/${CLUSTER_NAME}_pypoetry/*.toml; do
    sed -i "s/PAS2330/$PROJECT_NAME/g" "$file"
    sed -i "s/__RANDOM_SUFFIX_TO_BE_REPLACED_BY_INITIALIZE_POETRY_SH__/${RANDOM_SUFFIX}/g" "$file"
    sed -i "s/shadowluo/${USERNAME}/g" "$file"
done

source $HOME/.bashenv

module load $DEFAULT_PYTHON_VERSION
mkdir -p /fs/scratch/$PROJECT_NAME/${USERNAME}/envs

# 先删除所有同名但不同后缀的环境, 通过从POETRY_ROOT_DIR中提取环境名实现
if [ -z "$POETRY_ROOT_DIR" ]; then
    echo "POETRY_ROOT_DIR is not set. Please check your .bashenv and bashprofile." >&2
    exit 1
fi
rm -rf "${POETRY_CLUSTER_DIR:?'POETRY_CLUSTER_DIR is empty!!!'}*"
rm -rf ~/.cache/pypoetry
python -m venv $POETRY_ROOT_DIR --upgrade-deps
source $POETRY_ROOT_DIR/bin/activate
python -m pip install --upgrade pip
python -m pip install poetry uv


poetry self add poetry-plugin-export
poetry config virtualenvs.path "${POETRY_ROOT_DIR}/venvs"
poetry source add --priority=supplemental coredumped https://pypi.coredumped.tech


# bash <(curl -s https://webdav.coredumped.tech/s/5E47CnbS8Y6qTja/download/initialize_poetry.sh)
# wget https://webdav.coredumped.tech/s/5E47CnbS8Y6qTja/download/initialize_poetry.sh -O initialize_poetry.sh

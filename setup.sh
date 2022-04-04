#!/usr/bin/env bash

# Repositories to clone
declare -a repositories=("openai/CLIP" "crowsonkb/guided-diffusion" "assafshocher/ResizeRight.git" \
                         "pytorch3d-lite.git" "MiDaS.git" "disco-diffusion.git" "latent-diffusion.git" \
                         "taming-transformers" "AdaBins.git")

for URL in "${repositories[@]}"
do
  echo Cloning "$URL"
  FOLDER=$(basename "$URL")
  if [ ! -d "$FOLDER" ] ; then
    git clone "$URL" "$FOLDER"
  fi
done

# Install modules
git install -e ./CLIP
git install -e ./guided-diffusion
git install -e ./taming-transformers

# APT packages
if [ "$(uname)" == "Darwin" ]; then
    brew install imagemagick
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    apt-get install -y imagemagick
fi


# Rename the MIDAS utils file (TODO: Why?)
mv ./MiDaS/utils.py ./MiDaS/midas_utils.py
cp ./disco-diffusion/disco_xform_utils.py disco_xform_utils.py

# Model path
mkdir models
mkdir pretrained
wget -O models/dpt_large-midas-2f21e586.pt https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt
wget -O models/AdaBins_nyu.pt https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt

cp models/AdaBins_nyu.pt pretrained/AdaBins_nyu.pt
# Make required directories
mkdir init_images
mkdir images_out

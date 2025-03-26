# ! bin/bash

if [ -f "/mnt/userspace/hejiabei_space/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/mnt/userspace/hejiabei_space/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/mnt/userspace/hejiabei_space/anaconda3/bin:$PATH"
fi
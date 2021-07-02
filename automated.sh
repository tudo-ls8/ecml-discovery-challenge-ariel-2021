#!/bin/bash

ARIEL_IMAGE="s876cnsm:5000/heppe/ariel"
GPU_STUFF="--device /dev/nvidia0 --device /dev/nvidia1 --device /dev/nvidiactl --device /dev/nvidia-uvm"

MNT_DATA="/rdata/s01b_ls8_000/datensaetze/ariel-space-mission-2021"

NAME_ADD="--name ${USER}-ariel"

RESOURCE_DEFAULT="-c 1 -m 10G"
RESOURCE_TRAIN="-c 4 -m 300G"

MNT_OPTS="-v ${MNT_DATA}:/mnt/data/"

OUT_DIR="$(date +'%m-%d-%Y')"

# Run firefox container for submission
docker run -it ${RESOURCE_DEFAULT} --name ${USER}-ariel-selenium -d -p 4444:4444 -v /dev/shm:/dev/shm selenium/standalone-firefox

# Run learning process
docker run -it --rm --name ${USER}-ariel-train ${RESOURCE_TRAIN} ${MNT_OPTS} ${GPU_STUFF} ${ARIEL_IMAGE} /usr/bin/python3 /mnt/data/ecml-discovery-challenge/main.py --seed $RANDOM --complete --predict /mnt/data/${OUT_DIR}

# TODO: Get host and port of line 4 and pass via --url
# Submit process
docker run --rm -it --name ${USER}-ariel-submit ${RESOURCE_DEFAULT} ${MNT_OPTS} ${ARIEL_IMAGE} /usr/bin/python3 /mnt/data/ecml-discovery-challenge/submit_data.py --username TheReturnOfBasel321 --password Gwkilab123 --input /mnt/data/${OUT_DIR}/pred.csv_45 --url "http://s876cn01:4444/wd/hub"

# docker run --rm -it --name ${USER}-ariel-submit -c 1 -m 10G -v /rdata/s01b_ls8_000/datensaetze/ariel-space-mission-2021:/mnt/data/ s876cnsm:5000/heppe/ariel /usr/bin/python3 /mnt/data/ecml-discovery-challenge/submit_data.py --username TheReturnOfBasel321 --password Gwkilab123 --input /mnt/data/pred.csv_45 --url 'http://s876cn01:4444/wd/hub'

# Send mail
docker run --rm -it --name ${USER}-ariel-score ${RESOURCE_DEFAULT} ${MNT_OPTS} ${ARIEL_IMAGE} /usr/bin/python3 /mnt/data/ecml-discovery-challenge/get_score.py --maillogin smluhepp --password HIDDEN69 --username TheReturnOfBasel321
# docker run --rm -it --name ${USER}-ariel-score -c 1 -m 10G -v /rdata/s01b_ls8_000/datensaetze/ariel-space-mission-2021:/mnt/data/ s876cnsm:5000/heppe/ariel /usr/bin/python3 /mnt/data/ecml-discovery-challenge/get_score.py --maillogin smluhepp --password ... --username TheReturnOfBasel321

docker stop ${USER}-ariel-selenium && docker rm ${USER}-ariel-selenium

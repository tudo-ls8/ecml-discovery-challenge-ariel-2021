#!/bin/bash
set -e
NAME="ariel" # default container name
RESOURCES="-c 4 -m 384g" # default resources allocated by each container
MNT_DATA="/rdata/s01b_ls8_000/datensaetze/ariel-space-mission-2021" # mount point

# find the name of the image (with or without prefix)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ -f "${SCRIPT_DIR}/.PUSH" ]; then
    IMAGE="$(cat "${SCRIPT_DIR}/.PUSH")"
elif [ -f "${SCRIPT_DIR}/.IMAGE" ]; then
    IMAGE="s876cnsm:5000/$(cat "${SCRIPT_DIR}/.IMAGE")"
else
    echo "ERROR: Could not find any Docker image. Run 'make' or 'make image' first!"
    exit 1
fi
# always print usage information
echo "Found the Docker image ${IMAGE}"
echo "| Usage: $0 [-r <resources>] [-n <name>] [enqueue|selenium|submit] [args...]"

# runtime arguments
ARGS=
while [ "$1" != "" ]; do
case "$1" in
    -r|--resources) # configure resources
        RESOURCES="$2"
        shift 2
        ;;
    -n|--name) # configure the container name
        NAME="$2"
        shift 2
        ;;
    enqueue) # add a container to the queue
        args="${@:2}"
        echo "| Resources: '$RESOURCES'"
        echo "| Name: ${USER}-${NAME}"
        echo "| Args: $args"
        docker create \
            --label queue \
            --label autohash \
            --env constraint:nodetype==gpu \
            --volume $MNT_DATA:/mnt/data \
            --volume /home/$USER:/mnt/home \
            --device /dev/nvidia0 \
            --device /dev/nvidia1 \
            --device /dev/nvidiactl \
            --device /dev/nvidia-uvm \
            --name "${USER}-${NAME}-" \
            --user "${USER}" \
            $RESOURCES \
            $IMAGE \
            $args  # pass additional arguments to the container entrypoint
        exit 0 # done
        ;;
    selenium) # run a standalone firefox container
        echo "| Name: ${USER}-ariel-selenium"
        read -p "Run this DETACHED container? [y|N] " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker create \
                --tty --interactive --rm --detach \
                --env constraint:node==s876cn01 \
                --volume /dev/shm:/dev/shm \
                --publish 4444:4444 \
                --name "${USER}-ariel-selenium" \
                --user "${USER}" \
                -c 1 -m 10g \
                selenium/standalone-firefox
        fi
        exit 0 # done
        ;;
    submit) # submit the latest predictions
        if [ -f "${SCRIPT_DIR}/.SECRET" ]; then
            SECRET="$(cat "${SCRIPT_DIR}/.SECRET")"
        else
            echo "ERROR: Could not find ${SCRIPT_DIR}/.SECRET"
            exit 1
        fi
        echo "| Name: ${USER}-ariel-submit"
        read -p "Run this INTERACTIVE container? [y|N] " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker create \
                --tty --interactive --rm \
                --volume $MNT_DATA:/mnt/data \
                --name "${USER}-ariel-submit" \
                --user "${USER}" \
                -c 1 -m 10g \
                $IMAGE \
                /usr/bin/python3 \
                    /mnt/data/ecml-discovery-challenge/submit_data.py \
                    --username TheReturnOfBasel321 \
                    --password "${SECRET}" \
                    --input /mnt/data/pred.csv_45 \
                    --url 'http://s876cn01:4444/wd/hub'
            docker start "${USER}-ariel-submit"
            docker attach "${USER}-ariel-submit"
        fi
        exit 0 # done
        ;;
    *) break ;;
esac
done

# start a single container (only available from gateway machine)
args="${@:1}"
echo "| Resources: '$RESOURCES'"
echo "| Name: ${USER}-${NAME}"
echo "| Args: $args"
read -p "Run this INTERACTIVE container? [y|N] " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker create \
        --tty --interactive --rm \
        --env constraint:node==s876gn03 \
        --volume $MNT_DATA:/mnt/data \
        --volume /home/$USER:/mnt/home \
        --device /dev/nvidia0 \
        --device /dev/nvidia1 \
        --device /dev/nvidiactl \
        --device /dev/nvidia-uvm \
        --name "${USER}-${NAME}" \
        --user "${USER}" \
        $RESOURCES \
        $IMAGE \
        $args # pass additional arguments to the container entrypoint
    docker start "${USER}-${NAME}"
    docker attach "${USER}-${NAME}"
fi

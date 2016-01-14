#! /bin/bash

## Ensure the binary exists:
if [ ! -x ./src/cymatic3d/cymatic3d ]; then
	echo -e "\nError: Cymatic3d has not yet been compiled.\nPlease run 'make'\n\n"
	exit 1
fi


## Run cymatic3d as high-priority if script was run as root
if [ $(whoami) == "root" ]; then
    nice --adjustment=-15 ./src/cymatic3d/cymatic3d
    exit
fi


## If not running as root user, give option to run at high-priority
echo -e "\nHigh priority processing looks better, but requires root password."
read -p "Do you wish to run Cymatic3D at high-priority? (Y/N) " yn
echo
case $yn in
    [Yy]* ) sudo nice --adjustment=-15 ./src/cymatic3d/cymatic3d ;;
    [Nn]* ) ./src/cymatic3d/cymatic3d ;;
    * ) echo "ERROR: Please answer Y or N.";;
esac

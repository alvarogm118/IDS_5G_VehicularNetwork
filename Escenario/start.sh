#!/bin/bash
echo "--Starting the Network..."
./init.sh

echo "--Waiting emu-vim to be ready..."
sleep 3
sudo ovs-vsctl show
sudo osm-check-vimemu

echo "--Starting the NS Instance..."
sudo osm ns-create --ns_name v5G-1 --nsd_name v5G --vim_account emu-vim
sudo osm ns-list
sudo docker ps | grep v5G

echo "--Creating the User Plane network..."
sudo vnx -f vnx/vnx_UserPlane.xml -t

echo "--Creating the External Network..."
sudo vnx -f vnx/vnx_ExternalNet_Internet.xml -t

echo "--Applying configurations..."
sleep 12
./v5G-1.sh
echo

echo "--Starting the IDS in the NWDAF instance..."
echo "--IDS ready"
sudo docker exec -ti mn.dc1_v5G-1-3-ubuntu-1 python3 IDS_REAL.py









#!/bin/bash
echo "--Destroying Docker containers and OSM instance..."
./v5G_destroy.sh v5G-1

echo "--Deleting the Home Network..."
sudo vnx -f vnx/vnx_UserPlane.xml -P

echo "--Deleting the External Network..."
sudo vnx -f vnx/vnx_ExternalNet_Internet.xml -P


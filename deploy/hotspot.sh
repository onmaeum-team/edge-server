#!/bin/bash
set -e
IFACE=${IFACE:-wlan0}
SSID=${SSID:-ONMAEUM}
PASS=${PASS:-onmaeum1234}
nmcli dev wifi hotspot ifname "$IFACE" ssid "$SSID" password "$PASS"
nmcli connection modify Hotspot connection.autoconnect yes
IP=$(ip -4 addr show "$IFACE" | awk '/inet /{print $2}' | cut -d/ -f1)
echo "핫스팟: $SSID / $PASS"
echo "앱 주소: http://$IP:8080"

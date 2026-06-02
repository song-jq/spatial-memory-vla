#!/usr/bin/env bash
set -euo pipefail

mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh

if [[ ! -f /etc/ssh/ssh_host_rsa_key ]]; then
    ssh-keygen -A
fi

echo "root:root123" | chpasswd

if [[ -f /ssh/authorized_keys ]]; then
    cp /ssh/authorized_keys /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

/usr/sbin/sshd

exec "$@"

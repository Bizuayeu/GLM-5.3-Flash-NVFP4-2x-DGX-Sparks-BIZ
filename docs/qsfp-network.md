# QSFP direct-link hands-on — NetworkManager

[日本語](qsfp-network.ja.md) · [Full setup](../SETUP.md)

Create a fixed IPv4 path between two hosts while retaining the management Wi-Fi default route. **Physical link, IP reachability, RoCE configuration and actual NCCL transport are separate checks. Completing this guide does not qualify TP=2.**

## 1. Open management SSH and record the baseline

On Windows, open PowerShell or a PowerShell tab in Windows Terminal. To use an existing SSH configuration, replace this illustrative path and host alias with your own:

```powershell
ssh -F 'C:\path\to\ssh_config' node-a
```

Without a configuration file, substitute the account and management host in `ssh USER@MANAGEMENT_HOST`. Verify a new host key through a trusted channel and never ignore a changed-key warning. Run `hostname` after connecting to confirm the target. Use `exit` to return to PowerShell; use another tab for the other host.

Keep management SSH open on each host. The commands below run **inside the remote Linux shell**, not directly in PowerShell:

```sh
hostname
nmcli --version
nmcli -f NAME,UUID,TYPE,DEVICE connection show
ip -brief link
ip -brief address
ip route show table all
ip rule show
rdma link show
ibdev2netdev
```

Record the management route, previous connection UUID, selected interface and existing autoconnect priority privately. Inspect a specific previous profile with `nmcli connection show uuid <UUID>` without secret-display options. Check that the proposed subnet does not overlap LAN, VPN, container or policy routes.

## 2. Connect the cable and identify the interfaces

Use a vendor-supported cable. Check connector orientation and seat it fully without force, following the hardware manual. If already connected, there is no need to unplug it for this check.

```sh
ip -brief link
rdma link show
ibdev2netdev
```

Look for Ethernet `LOWER_UP` and RDMA `ACTIVE` / `LINK_UP`. A physical Spark QSFP port maps to multiple logical interfaces, so one cable can bring up multiple interfaces. Names need not match across hosts; consult [NVIDIA's mapping](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html) and actual observations.

Select one Ethernet interface per host for this guide. Do not assign the same subnet to multiple newly active interfaces. Using the additional logical path requires separate validation.

## 3. Choose the values

These are **examples**. Keep actual host identities and site values in a private report.

| Item | Node A | Node B |
|---|---|---|
| Profile name | `glm53-qsfp` | `glm53-qsfp` |
| Fixed IPv4 | `10.53.0.1/30` | `10.53.0.2/30` |
| Ethernet interface | Observed on A | Observed on B |
| HCA / GID | Check after IPv4 assignment | Check after IPv4 assignment |

For this /30, `.0` is the network and `.3` the broadcast address; the two hosts use `.1` and `.2`. Configure no gateway or DNS. Initially keep the current MTU and record it.

`ipv4.never-default yes` excludes this connection from the IPv4 default route. Autoconnect applies when conditions permit; its priority selects among candidate profiles on the same device. It is not a routing priority and does not replace an already active profile, so explicitly activate the new profile initially. See [NetworkManager's property reference](https://networkmanager.pages.freedesktop.org/NetworkManager/NetworkManager/nm-settings-nmcli.html).

## 4. Save and activate one host at a time

An AI can inspect and record results; when administrator authentication is needed, the human runs `sudo` in their own SSH terminal. Do not send passwords to the AI or report. Sudo requests the remote Linux account's password; invisible password input is normal. A trailing `\` continues a Linux shell command: do not append spaces after it. Stop on errors before running the next command.

On node A, replace the interface with its **observed** value:

```sh
FABRIC_IF='REPLACE_WITH_OBSERVED_INTERFACE'
FABRIC_CIDR='10.53.0.1/30'
FABRIC_PEER='10.53.0.2'
FABRIC_PROFILE='glm53-qsfp'
nmcli -f NAME,UUID,DEVICE connection show
```

Only if this profile name is absent, create it once:

```sh
sudo nmcli connection add type ethernet \
  con-name "$FABRIC_PROFILE" ifname "$FABRIC_IF" \
  ipv4.method manual ipv4.addresses "$FABRIC_CIDR" \
  ipv4.never-default yes ipv6.method disabled \
  connection.autoconnect yes connection.autoconnect-priority 100
```

After successful creation:

```sh
sudo nmcli connection up "$FABRIC_PROFILE"
nmcli -f GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS,IP4.GATEWAY device show "$FABRIC_IF"
ip route get "$FABRIC_PEER"
ip route get 1.1.1.1
```

Verify the local address, intended peer interface and unchanged management route for external destinations. `ip route get` only queries routing; it does not test Internet reachability. The peer may not answer ping until its IPv4 is configured.

After checking A, repeat on B using B's observed `FABRIC_IF`, `FABRIC_CIDR='10.53.0.2/30'` and `FABRIC_PEER='10.53.0.1'`.

If a profile already exists, do not repeat `add`. Inspect its UUID and settings. Activate that UUID if correct; otherwise limit an authorized correction to `connection modify uuid <UUID> ...`. Resolve unknown profile ownership before modifying it.

## 5. Verify both directions and routing

On each host, in the same SSH shell with its own variables:

```sh
ip -4 address show dev "$FABRIC_IF"
ip route get "$FABRIC_PEER"
ping -I "$FABRIC_IF" -c 4 -W 2 "$FABRIC_PEER"
ip route get 1.1.1.1
nmcli -f connection.id,connection.uuid,connection.interface-name,connection.autoconnect,connection.autoconnect-priority,ipv4.method,ipv4.addresses,ipv4.gateway,ipv4.never-default,ipv6.method connection show "$FABRIC_PROFILE"
ip -details link show dev "$FABRIC_IF"
```

Require the correct local IP, direct peer route, 4/4 replies in both directions, original external management route and saved autoconnect settings. If external access is needed, separately check DNS resolution and reachability to a known HTTPS service you normally use.

Measure both MTUs before changing them. If both are 1500, test nonfragmented IPv4 ICMP with a 1472-byte payload:

```sh
ping -I "$FABRIC_IF" -M do -s 1472 -c 4 -W 2 "$FABRIC_PEER"
```

Do not reuse that size for a different MTU without checking. Raising MTU to 9000 is a separate change after confirming support along the path and arranging a change window. On the reference pair it bought at most 2–3% prefill for 1.1–1.7 GiB less free memory per host, so the pair stays at 1500 ([measurements](nccl-validation.md#channel-count)). Ping does not measure line speed or RDMA throughput.

## 6. Record RoCE mapping

Use `ibdev2netdev` to identify the HCA for the selected interface. Under `/sys/class/infiniband/<HCA>/ports/1/`, compare the same index in `gids/`, `gid_attrs/types/` and `gid_attrs/ndevs/`.

Select an index whose type is RoCE v2, net device is the selected interface, and IPv4-mapped GID matches the local fixed IPv4. Measure the index on each host; do not assume the commonly seen value `3`. Pass these observations to [site configuration](operations.md#network-and-site-configuration). An ACTIVE link or an available GID does not prove NCCL used that transport.

## 7. Persistence, recovery and completion

Saved profile/autoconnect settings are not a completed reboot test. Test an actual reboot only in a maintenance window after checking other jobs and management recovery, then repeat steps 5–6. This guide does not automatically reboot hosts.

On failure, save interface, profile, routes and carrier state. Do not begin recovery by restarting all of NetworkManager, changing Wi-Fi profiles or disabling the entire firewall. To roll back, use management access to disable autoconnect on the new UUID, deactivate it and reactivate the recorded previous profile:

```sh
sudo nmcli connection modify uuid <NEW_UUID> connection.autoconnect no
sudo nmcli connection down uuid <NEW_UUID>
sudo nmcli connection up uuid <PREVIOUS_UUID>
```

Replace angle-bracket placeholders with actual UUIDs. Omit the last command if no previous profile existed. Preserve profiles instead of deleting evidence.

- [ ] Supported cable connected; physical link confirmed at both ends.
- [ ] Management path and previous profiles recorded; subnet conflicts checked.
- [ ] Correct fixed IPv4, never-default and autoconnect saved per host.
- [ ] Bidirectional ping, routes and MTU checked.
- [ ] HCA and IPv4-mapped RoCEv2 GID recorded.
- [ ] Reboot persistence tested, or explicitly marked not run.
- [ ] Actual NCCL transport, bandwidth and TP=2 inference handed off as separate pending checks.

Record steps, operator, timestamps/timezone, results, evidence paths and next actions privately in `records/<run-id>/REPORT.md`.

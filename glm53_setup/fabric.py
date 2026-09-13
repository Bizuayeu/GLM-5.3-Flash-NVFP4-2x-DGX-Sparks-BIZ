"""Explicit RoCE rail configuration and read-only checks for every selected port."""

import ipaddress
import json
import re
from pathlib import Path


def rails(site):
    primary = {key: site[key] for key in ("hca", "interface", "local_ip", "gid_index")}
    primary["port"] = 1
    extra = site.get("additional_rails", [])
    if not isinstance(extra, list):
        raise ValueError("additional_rails must be a list of rail records")
    result = [primary, *extra]
    ports, interfaces, addresses = set(), set(), set()
    for rail in result:
        if not isinstance(rail, dict) or rail.keys() != primary.keys():
            raise ValueError(
                "Each rail requires hca, port, interface, local_ip, gid_index"
            )
        for key in ("hca", "interface"):
            if not isinstance(rail[key], str) or not re.fullmatch(
                r"[A-Za-z0-9_.-]+", rail[key]
            ):
                raise ValueError(f"Invalid rail {key}; use structured additional_rails")
        if rail["interface"].startswith("wl"):
            raise ValueError("RoCE rails cannot use Wi-Fi")
        if type(rail["port"]) is not int or rail["port"] < 1:
            raise ValueError("Rail port must be a positive integer")
        if (
            type(rail["gid_index"]) is not int
            or rail["gid_index"] != primary["gid_index"]
        ):
            raise ValueError("All rails must use the rank's common GID index")
        address = ipaddress.IPv4Address(rail["local_ip"])
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            raise ValueError("Rail address must be a fabric IPv4 address")
        identity = (rail["hca"], rail["port"])
        if identity in ports or rail["interface"] in interfaces or address in addresses:
            raise ValueError("Duplicate rail port, interface or address")
        ports.add(identity)
        interfaces.add(rail["interface"])
        addresses.add(address)
    return result


def checks(site, run, *, sys_root=Path("/sys"), dev_root=Path("/dev")):
    result = {"rdma_devices": (dev_root / "infiniband").is_dir()}

    def read(path):
        try:
            return path.read_text().strip()
        except (OSError, UnicodeError):
            return ""

    for index, rail in enumerate(rails(site)):
        net = sys_root / "class/net" / rail["interface"]
        port = sys_root / "class/infiniband" / rail["hca"] / "ports" / str(rail["port"])
        gid_index = str(rail["gid_index"])
        try:
            gid = ipaddress.IPv6Address(read(port / "gids" / gid_index))
        except ipaddress.AddressValueError:
            gid = None
        try:
            addresses = (
                json.loads(run("ip", "-j", "addr", "show", "dev", rail["interface"]))
                if net.exists()
                else []
            )
        except (OSError, ValueError):
            addresses = []
        prefix = f"rail_{index}_"
        result.update(
            {
                prefix + "link_up": read(net / "operstate") == "up",
                prefix + "port_active": read(port / "state").startswith("4:")
                and read(port / "phys_state").startswith("5:")
                and read(port / "link_layer") == "Ethernet",
                prefix + "roce_v2_gid": gid is not None
                and gid.ipv4_mapped == ipaddress.IPv4Address(rail["local_ip"])
                and read(port / "gid_attrs/types" / gid_index) == "RoCE v2"
                and read(port / "gid_attrs/ndevs" / gid_index) == rail["interface"],
                prefix + "address_assigned": any(
                    a.get("local") == rail["local_ip"]
                    for n in addresses
                    for a in n.get("addr_info", [])
                ),
            }
        )
    return result

"""A rank's site and explicit RoCE rails: their validation, the NCCL environment they
imply, and read-only checks for every selected port. Two nodes share one link (plus
optional rails); three or more are cabled as a switchless ring, each node listing its
direct link to every other node."""

import ipaddress
import json
import re
from pathlib import Path

# A device or interface name taken from the host's inventory.
DEVICE_NAME = r"[A-Za-z0-9_.-]+"


def is_fabric_ipv4(value):
    """Whether an IPv4 address can carry fabric traffic (not loopback, 0.0.0.0 or multicast)."""
    address = ipaddress.IPv4Address(value)
    return not (address.is_loopback or address.is_unspecified or address.is_multicast)


def is_wifi(interface):
    return interface.startswith("wl")


def rail_check(index, name):
    """The key of one rail's check in checks(); gid_hints reads the same key."""
    return f"rail_{index}_{name}"


# What a ring node writes for each direct link; the port is always 1.
LINK_KEYS = frozenset({"peer", "hca", "interface", "local_ip", "peer_ip", "gid_index"})
RAIL_KEYS = ("hca", "interface", "local_ip", "gid_index")


def rails(site):
    if "links" in site:
        # A ring node's rails are its links, in the order it lists them.
        result = [
            {**{key: link[key] for key in RAIL_KEYS}, "port": 1}
            for link in site["links"]
        ]
        primary = result[0]
    else:
        primary = {key: site[key] for key in RAIL_KEYS}
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
                DEVICE_NAME, rail[key]
            ):
                raise ValueError(f"Invalid rail {key}; use structured additional_rails")
        if is_wifi(rail["interface"]):
            raise ValueError("RoCE rails cannot use Wi-Fi")
        if type(rail["port"]) is not int or rail["port"] < 1:
            raise ValueError("Rail port must be a positive integer")
        if (
            type(rail["gid_index"]) is not int
            or rail["gid_index"] < 0
            or rail["gid_index"] != primary["gid_index"]
        ):
            raise ValueError("All rails must use the rank's common GID index")
        if not is_fabric_ipv4(rail["local_ip"]):
            raise ValueError("Rail address must be a fabric IPv4 address")
        address = ipaddress.IPv4Address(rail["local_ip"])
        identity = (rail["hca"], rail["port"])
        if identity in ports or rail["interface"] in interfaces or address in addresses:
            raise ValueError("Duplicate rail port, interface or address")
        ports.add(identity)
        interfaces.add(rail["interface"])
        addresses.add(address)
    return result


def node_count(site):
    """How many nodes the launch has; a two-node site without links does not say."""
    return site.get("nnodes", 2)


def check_rank(rank, nodes):
    if type(rank) is not int or not 0 <= rank < nodes:
        raise ValueError(f"rank must be from 0 to {nodes - 1}")


def validate_site(site):
    check_rank(site.get("rank"), node_count(site))
    for key in ("head_ip", "local_ip"):
        if not is_fabric_ipv4(site[key]):
            raise ValueError(f"{key} must be a fabric IPv4 address")
    if (site["rank"] == 0) != (site["head_ip"] == site["local_ip"]):
        raise ValueError("Head must own head_ip; worker must have a different address")
    # A ring site's HCAs and GID index are its links'; its interface is the socket's.
    for key in ("interface",) if "links" in site else ("interface", "hca"):
        if not re.fullmatch(DEVICE_NAME, site.get(key, "")):
            raise ValueError(f"Set a concrete {key} from the local device inventory")
    # A ring's data path is its links (checked in rails); its socket interface may be
    # the management network, a test setting until the hosts carry a /32 (plan Stage 4).
    if "links" not in site and is_wifi(site["interface"]):
        raise ValueError("The real-model profile requires RoCE, not Wi-Fi")
    if "links" not in site and (
        type(site.get("gid_index")) is not int or site["gid_index"] < 0
    ):
        raise ValueError("gid_index must be nonnegative")
    for key in ("api_port", "master_port"):
        if type(site.get(key)) is not int or not 1024 <= site[key] <= 65535:
            raise ValueError(f"Invalid {key}")
    if site["api_port"] == site["master_port"]:
        raise ValueError("API and rendezvous ports must differ")
    rails(site)


def validate_nodes(nodes):
    """Two nodes on one link, or every pair of two or more nodes on its own link.

    Every node lists its links or none does; three or more nodes need them. A link
    names the other node by rank, and the two ends of a link name each other's
    addresses inside one /30. A node's links share one GID index (NCCL takes one per
    rank). host_address/host_interface give a node one stable address (addressing).
    """
    linked = [isinstance(node, dict) and "links" in node for node in nodes]
    if any(linked) != all(linked):
        raise ValueError("Every node lists its links, or none does")
    if not any(linked):
        if len(nodes) != 2:
            raise ValueError(
                "Three or more nodes need links: one direct link per pair of nodes"
            )
        if any("host_address" in n or "host_interface" in n for n in nodes):
            raise ValueError("host_address belongs to nodes with links")
        return
    for rank, node in enumerate(nodes):
        name = f"nodes[{rank}]"
        if "additional_rails" in node:
            raise ValueError(
                f"{name} takes its rails from its links; remove additional_rails"
            )
        links = node["links"]
        if (
            not isinstance(links, list)
            or not links
            or any(not isinstance(i, dict) or i.keys() != LINK_KEYS for i in links)
        ):
            raise ValueError(
                f"{name}.links: each link needs peer, hca, interface, local_ip, "
                "peer_ip, gid_index"
            )
        peers = [link["peer"] for link in links]
        if any(type(peer) is not int for peer in peers) or sorted(peers) != [
            peer for peer in range(len(nodes)) if peer != rank
        ]:
            raise ValueError(f"{name}.links must name every other node once")
        for link in links:
            if not isinstance(link["peer_ip"], str) or not is_fabric_ipv4(
                link["peer_ip"]
            ):
                raise ValueError(f"{name}.links: peer_ip must be a fabric IPv4 address")
        rails({"links": links})
        if ("host_address" in node) != ("host_interface" in node):
            raise ValueError(f"{name} needs host_address and host_interface together")
        if "host_interface" in node and (
            not isinstance(node["host_interface"], str)
            or not re.fullmatch(DEVICE_NAME, node["host_interface"])
        ):
            raise ValueError(f"{name}.host_interface must name one interface")
    link_addresses = set()
    for rank, node in enumerate(nodes):
        for link in node["links"]:
            back = link_toward(nodes[link["peer"]], rank)
            ends = (link["local_ip"], link["peer_ip"])
            if ends != (back["peer_ip"], back["local_ip"]) or not one_slash30(*ends):
                raise ValueError(
                    f"The link between nodes {rank} and {link['peer']}: both ends "
                    "must name each other's addresses, the two hosts of one /30"
                )
            link_addresses.add(ipaddress.IPv4Address(link["local_ip"]))
    hosts = [node["host_address"] for node in nodes if "host_address" in node]
    for address in hosts:
        if not isinstance(address, str) or not is_fabric_ipv4(address):
            raise ValueError("host_address must be a fabric IPv4 address")
    addresses = [ipaddress.IPv4Address(a) for a in hosts]
    if len(set(addresses)) != len(addresses) or link_addresses & set(addresses):
        raise ValueError("Each host_address must be unique and no link's address")


def one_slash30(first, second):
    """Whether two addresses are the two host addresses of one /30."""
    network = ipaddress.IPv4Interface(f"{first}/30").network
    return first != second and {
        ipaddress.IPv4Address(first),
        ipaddress.IPv4Address(second),
    } <= set(network.hosts())


def link_toward(node, peer):
    return next(link for link in node["links"] if link["peer"] == peer)


def addressing(nodes, rank):
    """Where a ring rank meets the head's store, and what it advertises to the others.

    With host_address a rank advertises that address on host_interface, and every rank
    meets the head at the head's host_address. Without, a rank meets the head at the
    head's end of their direct link and advertises its own end (the head: its link to
    rank 1). Each rank picks for itself, so the two forms may be mixed.

    cc-defer: provisional. The fallback reaches the master from every rank, but Gloo
    pairs, NCCL's bootstrap ring and vLLM's broadcast queue connect every rank to each
    address advertised once, and a /30 link-to-head address is not reachable from the
    third node without routes; host_address needs a /32 on its own interface and
    static routes over the direct links. Plan Stage 3 sets up and tests the /32 and
    its routes on the ring, and settles which form the launcher keeps.
    """
    head, node = nodes[0], nodes[rank]
    if "host_address" in head:
        master = head["host_address"]
    else:
        master = link_toward(head, rank or 1)["local_ip"]
    if "host_address" in node:
        own, interface = node["host_address"], node["host_interface"]
    else:
        toward = link_toward(node, 1 if rank == 0 else 0)
        own, interface = toward["local_ip"], toward["interface"]
    return {"head_ip": master, "local_ip": own, "interface": interface}


def fabric_env(site):
    validate_site(site)
    selected = rails(site)
    hcas = ",".join(f"{r['hca']}:{r['port']}" for r in selected)
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_HOST_IP": site["local_ip"],
        "NCCL_NET": "IB",
        "NCCL_IB_DISABLE": "0",
        "NCCL_SOCKET_IFNAME": "=" + site["interface"],
        "GLOO_SOCKET_IFNAME": site["interface"],
        "NCCL_IB_HCA": "=" + hcas,
        "NCCL_IB_GID_INDEX": str(selected[0]["gid_index"]),
        "NCCL_IB_ROCE_VERSION_NUM": "2",
        "NCCL_IB_ADDR_FAMILY": "AF_INET",
        "NCCL_DEBUG": "INFO",
    }
    if node_count(site) >= 3:
        # NVIDIA's three-Spark ring: NCCL 2.30.7 sends to each peer over the link
        # whose subnet reaches it (2.29.7 lacks the knob).
        # cc-defer: set without checking that the image's NCCL knows the knob;
        # preflight reads no NCCL version and the image carries no marker for it.
        # Plan Stage 3 confirms it in the probe's NCCL log on the ring.
        env["NCCL_IB_SUBNET_AWARE_ROUTING"] = "1"
    return env


def link_probes(site):
    """For a ring site, how to run the two-rank NCCL probe over each of its links.

    The lower rank of the pair is the probe's rank 0 and its end of the link the
    probe's head; the environment names only that link. None without links.
    """
    if "links" not in site:
        return None
    probes = []
    for link in site["links"]:
        lower = site["rank"] < link["peer"]
        head = link["local_ip"] if lower else link["peer_ip"]
        pair = {
            "rank": 0 if lower else 1,
            "head_ip": head,
            "local_ip": link["local_ip"],
            "interface": link["interface"],
            "hca": link["hca"],
            "gid_index": link["gid_index"],
            "api_port": site["api_port"],
            "master_port": site["master_port"],
        }
        probes.append(
            {
                "peer": link["peer"],
                "probe_rank": pair["rank"],
                "head": head,
                "environment": {**fabric_env(pair), "NCCL_SOCKET_FAMILY": "AF_INET"},
            }
        )
    return probes


def read(path):
    try:
        return path.read_text().strip()
    except (OSError, UnicodeError):
        return ""


def roce_v2_gid(port, index, rail):
    """Whether GID index on port is the rail's IPv4-mapped RoCE v2 entry."""
    try:
        gid = ipaddress.IPv6Address(read(port / "gids" / str(index)))
    except ipaddress.AddressValueError:
        return False
    return (
        gid.ipv4_mapped == ipaddress.IPv4Address(rail["local_ip"])
        and read(port / "gid_attrs/types" / str(index)) == "RoCE v2"
        and read(port / "gid_attrs/ndevs" / str(index)) == rail["interface"]
    )


def port_path(sys_root, rail):
    return sys_root / "class/infiniband" / rail["hca"] / "ports" / str(rail["port"])


def gid_hints(site, checks, *, sys_root=Path("/sys")):
    """Where the refused rails' RoCE v2 GIDs are now, for the operator to act on.

    A link that went down and came back can leave the entry at another index
    (upstream MiaAI-Lab recipe #277; our 2026-09-27 incident). The check still
    refuses: NCCL takes one index per rank, so the fix is a config or host change.
    """
    hints = []
    for number, rail in enumerate(rails(site)):
        if checks.get(rail_check(number, "roce_v2_gid"), True):
            continue
        port = port_path(sys_root, rail)
        try:
            indices = sorted(
                int(p.name) for p in (port / "gids").iterdir() if p.name.isdigit()
            )
        except OSError:
            indices = []
        hints.append(
            {
                "rail": number,
                "hca": rail["hca"],
                "port": rail["port"],
                "local_ip": rail["local_ip"],
                "configured_gid_index": rail["gid_index"],
                "roce_v2_gid_indices": [
                    i for i in indices if roce_v2_gid(port, i, rail)
                ],
            }
        )
    return hints


def assigned(run, sys_root, interface, address):
    """Whether the interface exists and carries the IPv4 address."""
    try:
        addresses = (
            json.loads(run("ip", "-j", "addr", "show", "dev", interface))
            if (sys_root / "class/net" / interface).exists()
            else []
        )
    except (OSError, ValueError):
        addresses = []
    return any(
        a.get("local") == address for n in addresses for a in n.get("addr_info", [])
    )


def checks(site, run, *, sys_root=Path("/sys"), dev_root=Path("/dev")):
    result = {"rdma_devices": (dev_root / "infiniband").is_dir()}

    for index, rail in enumerate(rails(site)):
        net = sys_root / "class/net" / rail["interface"]
        port = port_path(sys_root, rail)
        result.update(
            {
                rail_check(index, "link_up"): read(net / "operstate") == "up",
                rail_check(index, "port_active"): read(port / "state").startswith("4:")
                and read(port / "phys_state").startswith("5:")
                and read(port / "link_layer") == "Ethernet",
                rail_check(index, "roce_v2_gid"): roce_v2_gid(
                    port, rail["gid_index"], rail
                ),
                rail_check(index, "address_assigned"): assigned(
                    run, sys_root, rail["interface"], rail["local_ip"]
                ),
            }
        )
    if "host_address" in site:
        result["host_address_assigned"] = assigned(
            run, sys_root, site["host_interface"], site["host_address"]
        )
    return result

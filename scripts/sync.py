"""
Syncs all semprobe data between two machines.

Must run on the machine with remote access permission.
"""

import beartype
import tyro


@beartype.beartype
def from_remote(
    ssh_host: str = "strawberry0",
    remote_path: str = "~/projects/saev/data/semprobe/test",
    local_path: str = "./data/semprobe/test",
):
    """
    Syncs all data from ssh_host:remote_path to local_path using rsync or scp, depending on what is available on your system.

    Args:
        ssh_host:
        remote_path:
        local_path:
    """
    # Fill in this docstring. AI!
    pass


@beartype.beartype
def to_remote():
    pass


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "to-remote": from_remote,
        "from-remote": to_remote,
    })

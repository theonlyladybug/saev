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
        ssh_host: The hostname or IP address of the remote machine to sync from. Can be a user@host, or a HostName found in your .ssh/config file.
        remote_path: The path on the remote machine containing the data to sync
        local_path: The local destination path where data will be copied to
    """
    # Write this function. AI!


@beartype.beartype
def to_remote():
    pass


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "to-remote": to_remote,
        "from-remote": from_remote,
    })

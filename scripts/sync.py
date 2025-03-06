"""
Syncs all semprobe data between two machines.

Must run on the machine with remote access permission.
"""

import os
import shutil
import subprocess

import beartype
import tyro


@beartype.beartype
def from_remote(
    ssh_host: str = "strawberry0",
    remote_path: str = "~/projects/saev/data/contrib/semprobe/test",
    local_path: str = "./data/semprobe/test",
):
    """
    Syncs all data from ssh_host:remote_path to local_path using rsync or scp, depending on what is available on your system.

    Args:
        ssh_host: The hostname or IP address of the remote machine to sync from. Can be a user@host, or a HostName found in your .ssh/config file.
        remote_path: The path on the remote machine containing the data to sync
        local_path: The local destination path where data will be copied to
    """

    # Create local directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)

    # Try rsync first since it's more efficient
    if shutil.which("rsync"):
        cmd = [
            "rsync",
            "-avz",  # archive mode, verbose, compress
            "--progress",  # show progress
            "--include=*.jpg",  # include jpg
            "--include=*.jpeg",  # include jpeg
            "--include=*.png",  # include png
            "--include=*/",  # include directories
            "--exclude=*",  # exclude everything else
            "--exclude=.DS_Store",  # explicitly exclude DS_Store
            f"{ssh_host}:{remote_path}/",  # source with trailing slash
            f"{local_path}/",  # destination with trailing slash
        ]
    # Fall back to scp if rsync isn't available
    elif shutil.which("scp"):
        # For scp, we'll need to run multiple commands for each extension
        for ext in ["jpg", "jpeg", "png"]:
            cmd = [
                "scp",
                "-r",  # recursive
                f"{ssh_host}:{remote_path}/**/*.{ext}",  # source with glob
                local_path,  # destination
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                # Ignore errors from no matches for an extension
                pass
        return  # Return early since we've already handled the scp case
    else:
        raise RuntimeError("Neither rsync nor scp found in PATH")

    # Execute the sync command
    subprocess.run(cmd, check=True)


@beartype.beartype
def to_remote(
    ssh_host: str = "strawberry0",
    remote_path: str = "~/projects/saev/data/contrib/semprobe/test",
    local_path: str = "./data/semprobe/test",
):
    """
    Syncs all data from local_path to ssh_host:remote_path using rsync or scp, depending on what is available on your system.

    Args:
        ssh_host: The hostname or IP address of the remote machine to sync to. Can be a user@host, or a HostName found in your .ssh/config file.
        remote_path: The destination path on the remote machine where data will be copied to.
        local_path: The local source path with the data to sync
    """
    # Ensure local path exists and is readable
    if not os.path.exists(local_path):
        raise RuntimeError(f"Local path does not exist: {local_path}")

    # Create remote directory first
    mkdir_cmd = ["ssh", ssh_host, f"mkdir -p {remote_path}"]
    subprocess.run(mkdir_cmd, check=True)

    # Expand ~ in remote path if present
    if remote_path.startswith("~"):
        get_home_cmd = ["ssh", ssh_host, "echo $HOME"]
        home = subprocess.check_output(get_home_cmd).decode().strip()
        remote_path = remote_path.replace("~", home)

    # Try rsync first since it's more efficient
    if shutil.which("rsync"):
        cmd = [
            "rsync",
            "-avz",  # archive mode, verbose, compress
            "--progress",  # show progress
            "--include=*.jpg",  # include jpg
            "--include=*.jpeg",  # include jpeg
            "--include=*.png",  # include png
            "--include=*/",  # include directories
            "--exclude=*",  # exclude everything else
            "--exclude=.DS_Store",  # explicitly exclude DS_Store
            f"{local_path}/",  # source with trailing slash
            f"{ssh_host}:{remote_path}/",  # destination with trailing slash
        ]
    # Fall back to scp if rsync isn't available
    elif shutil.which("scp"):
        # For scp, we'll need to run multiple commands for each extension
        for ext in ["jpg", "jpeg", "png"]:
            cmd = [
                "scp",
                "-r",  # recursive
                f"{local_path}/**/*.{ext}",  # source with glob
                f"{ssh_host}:{remote_path}",  # destination
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                # Ignore errors from no matches for an extension
                pass
        return  # Return early since we've already handled the scp case
    else:
        raise RuntimeError("Neither rsync nor scp found in PATH")

    # Execute the sync command
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "to-remote": to_remote,
        "from-remote": from_remote,
    })

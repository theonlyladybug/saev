def main(pkg_names: list[str], fpath: str):
    import pdoc

    """
    Generate pdoc3 documentation and combine into a single text file.

    Args:
        pkg_names: Names of the Python package to document
        fpath: Path for output .txt file
    """

    # Function to process module and its submodules recursively
    def process_module(module) -> list[str]:
        # breakpoint()
        content = []
        # Add module content
        content.append(module.text())

        # Process submodules
        for submodule in module.submodules():
            content.extend(process_module(submodule))

        return content

    content = []
    for name in pkg_names:
        mod = pdoc.Module(name)
        content.extend(process_module(mod))

    # Process all modules and write to file
    with open(fpath, "w") as f:
        f.write("\n\n".join(content))


if __name__ == "__main__":
    import tyro

    tyro.cli(main)

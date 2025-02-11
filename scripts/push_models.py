repo_url = "https://github.com/OSU-NLP-Group/SAE-V"
docs_url = "https://osu-nlp-group.github.io/SAE-V"


def main(
    hf_token: str,
    folder: str = "checkpoints/public/usvhngx4",
    repo: str = "samuelstevens/SAE-CLIP-24K-ViT-B-16",
):
    import huggingface_hub as hfhub

    hfapi = hfhub.HfApi(token=hf_token)

    user, name = repo.split("/")

    hfapi.upload_folder(folder_path=folder, repo_id=repo, repo_type="model")
    hfapi.upload_file(
        path_or_fileobj=f"docs/modelcards/{name}.md",
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)

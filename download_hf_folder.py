from huggingface_hub import snapshot_download

# Downloads the repository from the namespace "amuvarma/test_ptjoints" to the "my_joints_data" directory
repo_dir = snapshot_download(repo_id="amuvarma/video_sapiens", local_dir="vids")
print(f"Repository downloaded to: {repo_dir}")

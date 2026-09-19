"""Optional benchmark downloader."""
import os
from src.paths import project_path
from huggingface_hub import snapshot_download

BASE_DIR = str(project_path(os.environ.get(
    "BASE_DIR",
    "data/General_Task_Benchmark"
)))

# (repo_id, subdir_name, repo_type, description)
DATASETS = [

    ("gaia-benchmark/GAIA", "GAIA", "dataset",
     "General agent tasks, 466 questions, 3 difficulty levels"),


    ("openai/openai_humaneval", "HumanEval", "dataset",
     "164 Python function-generation tasks"),


    ("HuggingFaceH4/MATH-500", "MATH-500", "dataset",
     "500-question MATH subset"),


    ("openai/gsm8k", "GSM8K", "dataset",
     "8500 grade-school math word problems"),


    ("princeton-nlp/SWE-bench_Verified", "SWE-Bench_Verified", "dataset",
     "500 verified code-repair tasks"),


    ("hotpotqa/hotpot_qa", "HotpotQA", "dataset",
     "Multi-hop QA and cross-document reasoning"),
]

def main():
    print(f"Base directory: {BASE_DIR}")
    os.makedirs(BASE_DIR, exist_ok=True)

    succeeded = []
    failed = []

    for repo_id, subdir, repo_type, desc in DATASETS:
        local_dir = os.path.join(BASE_DIR, subdir)
        print(f"\n{'='*60}")
        print(f"Downloading: {repo_id}")
        print(f"Description: {desc}")
        print(f"Target:      {local_dir}")
        print(f"{'='*60}")

        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=local_dir,
            )
            print(f"✓ {repo_id} done")
            succeeded.append(repo_id)
        except Exception as e:
            print(f"✗ {repo_id} FAILED: {e}")
            failed.append((repo_id, str(e)))

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Succeeded ({len(succeeded)}):")
    for r in succeeded:
        print(f"  ✓ {r}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r, err in failed:
            print(f"  ✗ {r}")
            print(f"    Reason: {err}")
        print("\nTroubleshooting:")
        print("  - GAIA: accept dataset terms and enable token read access")
        print("  - Network: check connectivity and the Hugging Face endpoint")
        print("  - Proxy: install httpx[socks]")

if __name__ == "__main__":
    main()

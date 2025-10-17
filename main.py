import modal
import os

app = modal.App("music-generator")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .uv_sync()
    .run_commands(
        [
            "git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step",
            "cd /tmp/ACE-Step && pip install .",
        ]
    )
    .env({"HF_HOME": "/.cache/huggingface"})
    .add_local_python_source("prompts")
)

# for open-source model
model_volume = modal.Volume.from_name(
    "ace-step-models", create_if_missing=True)
# for LLM
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

music_gen_secrets = modal.Secret.from_name("music-gen-secret")


@app.function(image=image, secrets=[modal.Secret.from_name("music-gen-secret")])
def function_test():
    print("Hello")
    print(os.environ["test"])


@app.local_entrypoint()
def main():
    function_test.remote()


if __name__ == "__main__":
    main()

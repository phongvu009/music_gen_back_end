import modal
from typing import List
import os
import base64
import uuid
import boto3

from pydantic import BaseModel
import requests

from prompts import PROMPT_GENERATOR_TEMPLATE, LYRICS_GENERATOR_TEMPLATE

app = modal.App("music-generator")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    # Ensure FFmpeg is available for torchcodec/libtorchcodec
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        [
            "git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step",
            "cd /tmp/ACE-Step && pip install -e .",
            # Print ffmpeg version during image build to aid diagnostics
            "ffmpeg -version || true",
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

#Create types - chema validation
class AudioGenerationBase(BaseModel):
    audio_duration: float = 60.0
    seed : int = -1
    guidance_scale: float = 15.0
    infer_step: int = 60

class GenerateFromDescriptionRequest(AudioGenerationBase):
    full_described_song: str

class GenerateWithCustomLyricsRequest(AudioGenerationBase):
    prompt: str
    lyrics: str

class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    prompt: str
    described_lyrics: str

class GenerateMusicResponseS3(BaseModel):
    s3_key: str
    cover_image_s3_key: str
    categories: List[str]

class GenerateMusicResponse(BaseModel):
    audio_data: str
    


#-------

@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=10
)

class MusicGenServer:
    @modal.enter()
    def load_model(self):
        # Debugging: show path info and attempt to ensure /tmp/ACE-Step is importable
        import sys
        import os
        print("--- import diagnostics: start ---")
        print("CWD:", os.getcwd())
        print("sys.executable:", sys.executable)
        print("sys.path:")
        for p in sys.path:
            print("  ", p)
        ace_tmp = "/tmp/ACE-Step"
        print(f"{ace_tmp} exists?", os.path.exists(ace_tmp))
        if os.path.exists(ace_tmp):
            try:
                print("/tmp/ACE-Step top-level:", os.listdir(ace_tmp))
            except Exception as e:
                print("unable to list /tmp/ACE-Step:", e)
        # If package wasn't installed, add the directory to sys.path as a fallback
        if ace_tmp not in sys.path:
            sys.path.insert(0, ace_tmp)
            print(f"Inserted {ace_tmp} into sys.path")
        print("--- import diagnostics: end ---")
        import importlib, subprocess, sys

        if importlib.util.find_spec("torchcodec") is None:
            print("torchcodec missing; installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torchcodec"])
            print("torchcodec installed")

        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from diffusers import AutoPipelineForText2Image
        import torch

        
        #Music Gen Model
        self.music_model = ACEStepPipeline(
            checkpoint_dir="/models",
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )
        
        # Large Language Model
        model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface"
        )

        # Stable Diffusion Model (thumbnails)
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", cache_dir="/.cache/huggingface")
        self.image_pipe.to("cuda")

    def prompt_qwen(self, question: str):
        messages=[
            {"role": "user", "content": question}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        generate_ids = self.llm.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )

        generate_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generate_ids)
        ]

        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

        return response

    #use LLM to generate prompt
    def generate_prompt(self, description: str) :
        full_prompt = PROMPT_GENERATOR_TEMPLATE.format(user_prompt=description)
        #send prompt to LLM
        return self.prompt_qwen(full_prompt)

    def generate_lyrics(self, description:str):
        full_prompt = LYRICS_GENERATOR_TEMPLATE.format(description=description)
        return self.prompt_qwen(full_prompt)
    
    def generate_categories(self, description:str) -> List[str]:
        prompt = f"Based on the following music description, list 3-5 relevant genres or categories as a comma-separated list. For example: Pop, Electronic, Sad, 80s. Description: '{description}'"

        response_text = self.prompt_qwen(prompt)
        categories = [ cat.strip() for cat in response_text.split(",") if cat.strip()]
        return categories

    
    def generate_and_upload_to_s3(
        self,
        prompt: str,
        lyrics: str,
        instrumental: bool,
        audio_duration: float,
        infer_step: int,
        guidance_scale: float,
        seed: int,
        description_for_categorization: str
    ) -> GenerateMusicResponseS3:
        final_lyrics = "[instrumental]" if instrumental else lyrics
        print(f"Generated lyrics:\n{final_lyrics}")
        print(f"Prompt:\n{prompt}")

        # S3 bucket: thumbnails , song
        # env setup in neon server
        s3_client = boto3.client("s3")
        bucket_name = os.environ("S3_BUCKET_NAME")

        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

        #create music
        self.music_model(
            prompt=prompt,
            lyrics=final_lyrics,
            audio_duration=audio_duration,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            save_path=output_path,
            manual_seed=str(seed) 
        )

        audio_s3_key = f"{uuid.uuid4()}.wav"
        s3_client.upload_file(output_path, bucket_name, audio_s3_key)
        #remove file after upload to s3
        os.remove(output_path)

        #generate thumbnail from text
        thumbnail_prompt = f"{prompt}, album cover art"
        image = self.image_pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        #save image on ne-on server
        image_output_path = os.path.join(output_dir, f"{uuid.uuid4()}.png")
        image.save(image_output_path)
        #upload to s3
        image_s3_key = f"{uuid.uuid4()}.png"
        s3_client.upload_file(image_output_path, bucket_name, image_s3_key)
        os.remove(image_output_path)

        #generate Category
        categories = self.generate_categories(description_for_categorization)
        
        return GenerateMusicResponseS3(
            s3_key=audio_s3_key,
            cover_image_s3_key=image_s3_key,
            categories=categories
        )


    @modal.fastapi_endpoint(method="POST")
    def generate(self) -> GenerateMusicResponse:
        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")
        
        try:
            #create music and saved 
            self.music_model(
                prompt="electronic rap",
                lyrics="[verse]\nWaves on the bass, pulsing in the speakers,\nTurn the dial up, we chasing six-figure features,\nGrinding on the beats, codes in the creases,\nDigital hustler, midnight in sneakers.\n\n[chorus]\nElectro vibes, hearts beat with the hum,\nUrban legends ride, we ain't ever numb,\nCircuits sparking live, tapping on the drum,\nLiving on the edge, never succumb.\n\n[verse]\nSynthesizers blaze, city lights a glow,\nRhythm in the haze, moving with the flow,\nSwagger on stage, energy to blow,\nFrom the blocks to the booth, you already know.\n\n[bridge]\nNight's electric, streets full of dreams,\nBass hits collective, bursting at seams,\nHustle perspective, all in the schemes,\nRise and reflective, ain't no in-betweens.\n\n[verse]\nVibin' with the crew, sync in the wire,\nGot the dance moves, fire in the attire,\nRhythm and blues, soul's our supplier,\nRun the digital zoo, higher and higher.\n\n[chorus]\nElectro vibes, hearts beat with the hum,\nUrban legends ride, we ain't ever numb,\nCircuits sparking live, tapping on the drum,\nLiving on the edge, never succumb.",
                audio_duration=120,
                infer_step=60,
                guidance_scale=15,
                save_path=output_path,
            )
    
            with open(output_path, "rb") as f:
                audio_bytes = f.read() # read audio file as bytes in memory
    
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8') # encode to base64 string
            
            return GenerateMusicResponse(audio_data=audio_base64)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)  # clean up temp file

    @modal.fastapi_endpoint(method="POST")
    def generate_from_description(self, request: GenerateFromDescriptionRequest) -> GenerateMusicResponse3:
        #
        prompt = self.generate_prompt(request.full_described_song)

        lyrics = ""

        #if not define as instrumental, create lyrics by AI based on song description
        if not request.instrumental:
            lyrics = self.generate_lyrics(request.full_described_song)
        #create song with lyrics
        return self.generate_and_upload_to_s3(
            prompt=prompt,
            lyrics=lyrics,
            description_for_categorization=request.full_described_song,
            **request.model_dump(exclude={"full_described_song"})
        )


    @modal.fastapi_endpoint(method="POST")
    def generate_with_lyrics(self, request: GenerateWithCustomLyricsRequest) -> GenerateMusicResponse3:
        return self.generate_and_upload_to_s3(
            prompt=request.prompt,
            lyrics=request.lyrics,
            description_for_categorization=request.prompt, #AI generate category based on prompt
            **request.model_dump() 
        )

    @modal.fastapi_endpoint(method="POST")
    def generate_with_described_lyrics(self, request: GenerateWithDescribedLyricsRequest) -> GenerateMusicResponse3:
        #Generate lyrics
        lyrics = ""
        if not request.instrumental:
            lyrics = self.generate_lyrics(request.described_lyrics)
        return self.generate_and_upload_to_s3(
            prompt=request.prompt,
            lyrics=lyrics,
            description_for_categorization=request.prompt, #AI generate category based on prompt
            **request.model_dump(exclude={"described_lyrics"}) 
        )


@app.local_entrypoint()
def main():
    server = MusicGenServer()
    endpoint_url = server.generate_from_description.get_web_url() # get the url from @modal.fastapi_endpoint(method="POST")

    request_data = GenerateFromDescriptionRequest(
        full_described_song="Acoustic Ballad",
        guidance_scale=7.5
    )
    
    payload = request_data.model_dump()
    
    response = requests.post(endpoint_url, json=payload)
    response.raise_for_status()
    result = GenerateMusicResponse(**response.json()) # validate response with pydantic model

    if result:
        print(f"Success: {result.s3_key} - {result.cover_image_s3_key} - {result.categories}")

    # audio_bytes = base64.b64decode(result.audio_data) # decode base64 string back to bytes
    # with open("generated_music.wav", "wb") as f:
    #     f.write(audio_bytes)  # write bytes to a file
    # print("Music generated and saved to generated_music.wav")
    


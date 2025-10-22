import boto3

s3 = boto3.client("s3")
s3.upload_file("generated_music.wav", "music-gen-test-bucket", "test_music.wav")
print("✅ Upload succeeded")

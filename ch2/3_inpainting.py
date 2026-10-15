from pathlib import Path

from diffusers import StableDiffusionInpaintPipeline
import torch
import PIL.Image as Image

device='cuda' if torch.cuda.is_available() else 'cpu'

pipe=StableDiffusionInpaintPipeline.from_pretrained('sd2-community/stable-diffusion-2-inpainting',torch_dtype=torch.float32).to(device)

img=Image.open('../data/p1ch2/horse.jpg')

mask_img=Image.open('../data/p1ch2/horse_mask.jpg')
prompt=('a zebra replacing the original horse, same pose, same lighting, background unchanged')
negative='distorted background, blurry, text, watermark'
out=pipe(prompt=prompt,image=img,mask_image=mask_img,negative_prompt=negative,guidance_scale=7.5,strength=0.8,generator=torch.Generator(device).manual_seed(42))
out.images[0].save(Path(__file__).parent/'3_inpainting_out.png')

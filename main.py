# Databricks notebook source
import inspect
import warnings
from typing import List, Optional, Union

import torch
from torch import autocast
from tqdm.auto import tqdm

from diffusers import StableDiffusionImg2ImgPipeline

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import textwrap
import random
import matplotlib.pyplot as plt

device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"
hf_token = "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"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=hf_token
)
pipe = pipe.to(device)


# COMMAND ----------

dbutils.widgets.text('prompt_input', '')
prompt_input = dbutils.widgets.get('prompt_input')

dbutils.widgets.text('prompt_image', '')
prompt_image = dbutils.widgets.get('prompt_image')

dbutils.widgets.text('logoUrl', '')
logoUrl = dbutils.widgets.get('logoUrl')

dbutils.widgets.text('promoCode', '')
promoCode = dbutils.widgets.get('promoCode')

dbutils.widgets.text('description', '')
description = dbutils.widgets.get('description')

dbutils.widgets.text('output_name', '')
output_name = dbutils.widgets.get('output_name')

# COMMAND ----------

import requests
from io import BytesIO
from PIL import Image

if len(prompt_image) > 0 :
    url = prompt_image
else :
    response = requests.get("https://pixabay.com/api/?key=30455955-0c53c333f0da10a9e19dedd44&q=" + prompt_input + "&image_type=photo&safesearch=true&order=popular&per_page=3")
    print(response.json()['hits'][0]["largeImageURL"])
    url = response.json()['hits'][0]["largeImageURL"]

# url = "https://pixabay.com/get/g0c81f85d9549f0e43dd42c6f05a24ec265a8c49d36037c9ecb6ea0159b6e5c0d8d996f15267006f8933cc5401cc4074cad7f7d71f558eed11240f500c1fdb071_1280.png"


print(url)
response = requests.get(url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
init_img = init_img.resize((768, 512))
init_img

# COMMAND ----------

generator = torch.Generator(device=device).manual_seed(39339852)
from diffusers import LMSDiscreteScheduler

lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
with autocast("cuda"):
#     image = pipe(prompt=prompt_input, init_image=init_img, strength=0.90, guidance_scale=7.5, generator=generator).images[0]
    image1 = pipe(prompt=prompt_input, init_image=init_img, strength=0.85, guidance_scale=7.5, generator=generator).images[0]
#     pipe.scheduler = lms
#     image2 = pipe(prompt=prompt_input, init_image=init_img, strength=0.75, guidance_scale=7.5, generator=generator).images[0]

# COMMAND ----------

import uuid

# image_generated_name = output_name + ".jpeg"
# image.save("/dbfs/mnt/hackathon/" + image_generated_name)

image_generated_name1 = str(uuid.uuid4()) + ".jpeg"
image1.save("/dbfs/mnt/hackathon/" + image_generated_name1)

# image_generated_name2 = str(uuid.uuid4()) + ".jpeg"
# image2.save("/dbfs/mnt/hackathon/" + image_generated_name2)


# image_generated_url = "https://fdpoc.blob.core.windows.net/ml-hackathon/" + image_generated_name
image_generated_url1 = "https://fdpoc.blob.core.windows.net/ml-hackathon/" + image_generated_name1
# image_generated_url2 = "https://fdpoc.blob.core.windows.net/ml-hackathon/" + image_generated_name2

# COMMAND ----------

plt.imshow(image1)
plt.show()

# COMMAND ----------

# from pathlib import Path

# from PIL import Image
# import matplotlib.pyplot as plt


# print(prompt)
# with autocast("cuda"):
#     images = pipe(prompt,strength=0.5, guidance_scale=105).images

# def display_image(image, dpi=100):
#     """
#     Description:
#         Displayes an image
#     Inputs:
#         path (str): File path
#         dpi (int): Your monitor's pixel density
#     """
#     img = image
#     width, height = img.size
#     plt.figure(figsize = (width/dpi,height/dpi))
#     plt.imshow(img, interpolation='nearest', aspect='auto')


# base_path = Path("/dbfs/hackathon")
# base_path.mkdir(exist_ok=True, parents=True)
# for i in range(len(images)):
#   image = images[i]
# #   image.save(base_path/f"test1.jpg")
#   display_image(image)

# COMMAND ----------

image_generated_url1

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Creative generation
# from PIL import Image, ImageDraw, ImageFont
# from pathlib import Path
# import textwrap
# import random
# import matplotlib.pyplot as plt

# COMMAND ----------

# load image components
bg = Image.open("/Workspace/Repos/manish.agarwal@inmobi.com/test/template/backgrounds/bg3.png")
bg = bg.resize((1728,1152))

phone = Image.open("/Workspace/Repos/manish.agarwal@inmobi.com/test/template/backgrounds/phone.png")
phone = phone.resize((500,1000))

# logo = Image.open("/Workspace/Repos/manish.agarwal@inmobi.com/test/template/logo_2.png")
# logo = logo.resize((300,300))

response_logo = requests.get(logoUrl)
logo = Image.open(BytesIO(response_logo.content)).convert("RGB")
logo = logo.resize((300, 300))

# app1 = Image.open("/Workspace/Repos/manish.agarwal@inmobi.com/test/template/backgrounds/app_1 (1).png")
# app1 = app.resize((300,100))

# gen_img = Image.open("/Workspace/Repos/manish.agarwal@inmobi.com/test/template/pizza.png")
gen_img = image1
gen_img = gen_img.resize((438,950))

# COMMAND ----------

# text inputs from user

text_desc = description
promo = promoCode
cta = "Download App"

# COMMAND ----------

def add_img(img_to_paste, base_img, x, y):
    back_im = base_img.copy()

    back_im.paste(img_to_paste, (x, y), img_to_paste)
    return back_im


def add_img_no_mask(img_to_paste, base_img, x, y):
    back_im = base_img.copy()

    back_im.paste(img_to_paste, (x, y))
    return back_im

def add_text_no_box(img, x, y, text="", font_type= "/Workspace/Repos/manish.agarwal@inmobi.com/test/template/Roboto-Bold.ttf", font_weight=1, font_size=50, char_width=15, font_fill=(255,255,255,255)):
    width, height = img.size

    lines = textwrap.wrap(text, width=char_width)

    font = ImageFont.truetype(font_type, size=font_size)

    draw = ImageDraw.Draw(img, "RGBA")

    y_text = y
    for line in lines:
        w, h = font.getsize(line)
        draw.text((x + w// 2, y_text), line, font=font, anchor='mm', fill=font_fill, stroke_width=font_weight)
        y_text += h

    return img


def add_text(img, x, y, text="", font_type= "/Workspace/Repos/manish.agarwal@inmobi.com/test/template/Roboto-Bold.ttf", font_weight=1,font_size=50, char_width=15, rect_fill=(0,0,0,100), font_fill=(255,255,255,255)):
    w_img, h_img = img.size

    lines = textwrap.wrap(text, width=char_width)
    font = ImageFont.truetype(font_type, size=font_size)
    draw = ImageDraw.Draw(img,"RGBA")

    num_lines = len(lines)
    if num_lines == 0 :
        return img

    w_r = 0
    h_r = 0

    for line in lines:
        w,h = font.getsize(line)
        w_r = max(w_r, w)
        h_r += 2*h
    w_r = (11*w_r)//10

    x1 = x
    y1 = y

    x2 = x1 + w_r
    y2 = y1 + h_r

    draw.rectangle((x1, y1, x2, y2), fill=rect_fill)

    h = h_r//num_lines

    X = (x1+x2)//2
    Y = y1+(h//2)
    for line in lines:
        draw.text((X,Y), line, font=font,anchor='mm', fill=font_fill, stroke_width=font_weight)
        Y += h

    return img

# COMMAND ----------

# add logo
bg_logo = add_img_no_mask(logo, bg, 1375, 78)

# add image
bg_logo_gen = add_img_no_mask(gen_img, bg_logo, 800, 100)

# add phone
bg_logo_gen_phone = add_img(phone, bg_logo_gen, 774, 76)

# add text desc
bg_logo_gen_phone_text = add_text_no_box(img = bg_logo_gen_phone, text = text_desc, x=160, y=270, font_weight=2, font_size=80, char_width=10)

# add promo
bg_logo_gen_phone_text_promo = add_text(img = bg_logo_gen_phone_text, text = promo, x=120, y=721, rect_fill=(255,57,33,255))

# add cta
bg_logo_gen_phone_text_promo_cta = add_text(img = bg_logo_gen_phone_text_promo, text = cta, x=120, y=880, rect_fill=(255,129,37,255))

# #add appstore and playstore logo
# bg_logo_gen_phone_text_promo_cta_app = add_img_no_mask(app1, bg_logo_gen_phone_text_promo_cta, 1420, 930)

# COMMAND ----------

def save_img(img, p_id):
    path = "/dbfs/mnt/hackathon/"
    final_str = path + output_name + "_output"+"_"+str(p_id)+".jpeg"
    img.save(final_str)
    image_generated_url = "https://fdpoc.blob.core.windows.net/ml-hackathon/" + output_name + "_output"+"_"+str(p_id)+".jpeg"
    print(image_generated_url)

# COMMAND ----------

template_count = 0
save_img(bg_logo_gen_phone_text_promo_cta, template_count)
template_count += 1

# COMMAND ----------

import requests
from io import BytesIO
from PIL import Image
import json


def process_tempate(user):
    url = "https://sync.api.bannerbear.com/v2/images"

#     headers_Aman = {"Authorization": "Bearer bb_pr_797bf8edd3fbb5a3588ba61f93a5ba" }
#     headers_Manish = {"Authorization": "Bearer bb_pr_8edf43ae4be7a5d80b83f07394bdc5" }

    templates = {
        "aman": {
            "key": "Bearer bb_pr_797bf8edd3fbb5a3588ba61f93a5ba",
               "data": [
                {
                    "id": "Aqa9wzDPloBNDJogk7",
                    "p_key": "0",
                    "name": "Instagram Carousel End Cover"
                },
                {
                    "id": "4KnlWBbK17QW5OQGgm",
                    "p_key": "1",
                    "name": "Health E-commerce Website Banner"
                }
               ]
        },
        "manish": {
            "key": "Bearer bb_pr_8edf43ae4be7a5d80b83f07394bdc5",
               "data": [
                {
                    "id": "lzw71BD606olZ0eYkn",
                    "p_key": "0",
                    "name": "Instagram Carousel End Cover"
                },
                {
                    "id": "p8YXW3b1xv73DVgkw2",
                    "p_key": "1",
                    "name": "Health E-commerce Website Banner"
                }
               ]
        },
        "kalyan": {
            "key": "Bearer bb_pr_0b17df050d5ae897f4ce8e779ef59c",
               "data": [
                {
                    "id": "Kp21rAZjG0xG56eLnd",
                    "p_key": "0",
                    "name": "Instagram Carousel End Cover"
                },
                {
                    "id": "RnxGpW5lj6xybEXrJ1",
                    "p_key": "1",
                    "name": "Health E-commerce Website Banner"
                }
               ]
        },
        "abhinav": {
            "key": "Bearer bb_pr_4a17c6f205643afa9f2a59a2f9fbd9",
               "data": [
                {
                    "id": "8BK3vWZJ2VklZJzk1a",
                    "p_key": "0",
                    "name": "Instagram Carousel End Cover"
                },
                {
                    "id": "wXmzGBDa1BPyDLN7gj",
                    "p_key": "1",
                    "name": "Health E-commerce Website Banner"
                }
               ]
        },
        "shashi": {
            "key": "Bearer bb_pr_f2951261d56fd85289ff52ebcc638f",
               "data": [
                {
                    "id": "agXkA3Dw3W8WDW2VBY",
                    "p_key": "0",
                    "name": "Instagram Carousel End Cover"
                },
                {
                    "id": "A37YJe5q19x05mpvWK",
                    "p_key": "1",
                    "name": "Health E-commerce Website Banner"
                }
               ]
        }
    }




#     templates_Manish = [
#         {
#             id: "yKBqAzZ9xeknbvMx36",
#             p_key: "0",
#             name: "Instagram Carousel End Cover"
#         },
#         {
#             id: "p8YXW3b1xv73DVgkw2",
#             p_key: "1",
#             name: "Health E-commerce Website Banner"
#         }

#     ]

    print(templates[user]["data"])

    data_0 = {
      "template": templates[user]["data"][0]["id"],
      "modifications": [
        {
          "name": "title",
          "text": description,
          "color": "#FFFFFF",
    #       "background": null
        },
        {
          "name": "subtitle",
          "text": promoCode,
          "color": "#FFFFFF",
    #       "background": null
        },
        {
          "name": "CTA",
          "text": "Download now!",
          "color": "#000000",
          "background": "#D7E022"
        },
    #     {
    #       "name": "username",
    #       "text": "@pizzahut",
    # #       "color": null,
    # #       "background": null
    #     },
        {
          "name": "product_image",
          "image_url": image_generated_url1,
    #       "fill_type": "fill"
        },
    #     {
    #       "name": "background",
    #       "image_url": image_generated_url,
    #       "fill_type": "fill"
    #     },
        {
         "name": "logo",
          "image_url": logoUrl,
    #       "fill_type": "fill"
        },
        {
          "name": "play_store",
          "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/2560px-Google_Play_Store_badge_EN.svg.png"
        },
        {
          "name": "apple_store",
          "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Download_on_the_App_Store_Badge.svg/2560px-Download_on_the_App_Store_Badge.svg.png"
        }
      ],
    #   "webhook_url": null,
      "transparent": "false",
    #   "metadata": null
    }


    data_1 = {
      "template": templates[user]["data"][1]["id"],
      "modifications": [
        {
          "name": "title",
          "text": description,
    #       "color": null,
    #       "background": null
        },
        {
          "name": "subtitle",
          "text": promoCode,
    #       "color": null,
    #       "background": null
        },
        {
          "name": "CTA",
          "text": "To know more, download now!",
    #       "color": null,
    #       "background": null
        },
        {
          "name": "product_image",
          "image_url": image_generated_url1
        },
        {
          "name": "logo",
          "image_url": logoUrl
        },
        {
          "name": "play_store",
          "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/2560px-Google_Play_Store_badge_EN.svg.png"
        },
        {
          "name": "apple_store",
          "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Download_on_the_App_Store_Badge.svg/2560px-Download_on_the_App_Store_Badge.svg.png"
        }
      ],
    #   "webhook_url": null,
      "transparent": "false",
    #   "metadata": null
    }

    headers = {"Authorization":  templates[user]["key"]}
    response = requests.post(url, headers=headers, json=data_0)
    response1 = requests.post(url, headers=headers, json=data_1)
    print(response.json())
    print(response1.json())
    return [response.json()["image_url"], response1.json()["image_url"]]

# COMMAND ----------

creatives = process_tempate("manish")

# COMMAND ----------

for creative in creatives:
    response = requests.get(creative)
    creative_img = Image.open(BytesIO(response.content)).convert("RGB")
    save_img(creative_img,template_count)
    template_count+=1

# COMMAND ----------



# COMMAND ----------

def resize_img(image, box_width, box_height):
    width, height = image.size
    ratio = min(box_height/height, box_width/width)
    new_height = (int)(ratio * height)
    new_width = (int)(ratio * width)
    image = image.resize((new_width, new_height))
    print("resizing logo..")
    return image

def add_text_bg(img, text):
    w_img, h_img = img.size

    lines = textwrap.wrap(text, width=15)
    font = ImageFont.truetype("/Workspace/Repos/manish.agarwal@inmobi.com/test/template/Roboto-Black.ttf", size=30)
    draw = ImageDraw.Draw(img,"RGBA")

    num_lines = len(lines)
    if num_lines == 0 :
        return img

    w_r = 0
    h_r = 0

    for line in lines:
        w,h = font.getsize(line)
        w_r = max(w_r, w)
        h_r += 2*h
    w_r = (11*w_r)//10
#     set coordinates
#     x1 can vary from 5% to 95% - w_r of width of image
#     y1 can vary from 5% to 75% - h_r of width of image

    x1 = random.randint((int)(0.05*w_img), max((int)(0.05*w_img),(int)(0.30*w_img - w_r)))
    y1 = random.randint((int)(0.05*h_img), max((int)(0.05*h_img),(int)(0.30*h_img - h_r)))

    x2 = x1 + w_r
    y2 = y1 + h_r

    draw.rectangle((x1, y1, x2, y2), fill=(0,0,0,100))

    h = h_r//num_lines

    X = (x1+x2)//2
    Y = y1+(h//2)
    for line in lines:
        draw.text((X,Y), line, font=font,anchor='mm', fill=(255,255,255,255), stroke_width=1)
        Y += h

    return img

def add_promo_bg(img, promo):
    w_img, h_img = img.size

    lines = textwrap.wrap(promo, width=10)
    font = ImageFont.truetype("/Workspace/Repos/manish.agarwal@inmobi.com/test/template/Roboto-Bold.ttf", size=26)
    draw = ImageDraw.Draw(img,"RGBA")

    num_lines = len(lines)
    if num_lines == 0 :
        return img

    w_r = 0
    h_r = 0

    for line in lines:
        w,h = font.getsize(line)
        w_r = max(w_r, w)
        h_r += 2*h
    w_r = (11*w_r)//10

    x1 = (int)(0.05*w_img)
    y1 = (int)(0.75*h_img - h_r)

    x2 = x1 + w_r
    y2 = y1 + h_r

    draw.rectangle((x1, y1, x2, y2), fill=(255,255,255,100))

    h = h_r//num_lines

    X = (x1+x2)//2
    Y = y1+(h//2)
    for line in lines:
        draw.text((X,Y), line, font=font,anchor='mm', fill=(0,0,0,255), stroke_width=1)
        Y += h

    return img

def add_cta_bg(img, cta):
    w_img, h_img = img.size

    lines = textwrap.wrap(cta, width=15)
    font = ImageFont.truetype("/Workspace/Repos/manish.agarwal@inmobi.com/test/template/Roboto-Black.ttf", size=26)
    draw = ImageDraw.Draw(img,"RGBA")

    num_lines = len(lines)
    if num_lines == 0 :
        return img

    w_r = 0
    h_r = 0

    for line in lines:
        w,h = font.getsize(line)
        w_r = max(w_r, w)
        h_r += 2*h
    w_r = (11*w_r)//10

    x1 = (int)(0.98*w_img - w_r)
    y1 = (int)(0.60*h_img)

    x2 = x1 + w_r
    y2 = y1 + h_r

    draw.rounded_rectangle((x1, y1, x2, y2), radius=10,fill=(0,200,0,255))

    h = h_r//num_lines

    X = (x1+x2)//2
    Y = y1+(h//2)
    for line in lines:
        draw.text((X,Y), line, font=font,anchor='mm', fill=(255,255,255,255), stroke_width=1)
        Y += h

    return img


def add_logo_bg(logo, img_txt):
    width, height = img_txt.size
    back_im = img_txt.copy()
    draw = ImageDraw.Draw(back_im, "RGBA")
    w_l, h_l = logo.size
    x1 = (75*width)//100
    y1 = (75*height)//100
    draw.rectangle((x1, y1, x1+(11*w_l)//10, y1+(11*h_l)//10), fill='white', outline='black', width=1)
    back_im.paste(logo, (x1+(5*w_l)//100,y1+(5*h_l)//100))
    return back_im

# COMMAND ----------

def generate_creative(width, height, logo, gen_img, text, promo):
    img = gen_img

    #resize logo
    logo = resize_img(logo,(width*2)//10, (2*height)//10)

    #text
    img_with_txt = add_text_bg(img, text)

    #promocode
    img_with_promo = add_promo_bg(img_with_txt, promo)

    #cta
    img_with_cta = add_cta_bg(img_with_promo,cta)

    #final placement
    final_image = add_logo_bg(logo, img_with_promo)

    return final_image

# COMMAND ----------

response_logo = requests.get(logoUrl)
logo_file = Image.open(BytesIO(response_logo.content)).convert("RGB")

im = image1.copy()
logoIm = logo_file

# COMMAND ----------

final_img = generate_creative(im.size[0], im.size[1], logoIm, im, text_desc, promo)

save_img(final_img,template_count)
template_count+=1

plt.imshow(final_img)
plt.show()

# COMMAND ----------



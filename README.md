# ___***ControlNet-Colorful: add color to controlnet results***___

<a href='https://ControlNet-Colorful.github.io'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href='https://huggingface.co/ControlNet-Colorful'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
[![GitHub](https://img.shields.io/github/stars/fastisrealslow/ControlNet-Colorful?style=social)](https://github.com/fastisrealslow/ControlNet-Colorful/)


---


## Introduction

Stable Diffusion and ControlNet are so easy to use, a large number of users use Canny and Lineart to control the Stable Diffusion model to generate rich content. But in actual scenarios, users hope to control the color space of the generated image in addition to outline and content control.

We found that there is no tool on the market that can control the generated color more accurately. We proposed ControlNet-Colorful, which adds color control information to the basic model of ControlNet (canny/lineart), and can convert pre-trained text to images. The diffusion model implements color control functions.

Under the same model parameter size and inference speed, it can achieve equivalent or even better results than the official model. ControlNet-Colorful can generalize to other custom models fine-tuned from the same base model, as well as to controllable generation using existing controllable tools.

![arch](assets/page/color_control_AnythingV5_v5PrtREbd7a34ac0adbcd0a88aaf4164f18fe04.jpeg)

## Release
- [2024/1/12] 🔥 We release the code


## Installation

```
# install latest diffusers
pip install diffusers==0.22.1

# download the models
cd ControlNet-Colorful

# then you can use the annotator
```

## Download Models

you can download models from [here](https://huggingface.co/h94/IP-Adapter). To run the demo, you should also download the following models:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [ControlNet models](https://huggingface.co/lllyasviel)

## Demo
### color canny demo (revAnimated_v122)
![arch](assets/page/color_control_AnythingV5_v5PrtRE3e4dd839bd6ac3f4aac00ba648f7ba4f.jpeg)
![color_canny_demo](assets/page/color_control_revAnimated_v122b4ca24d2d158a4b673aa86276c4f98f2.jpeg)
![color_canny_demo](assets/page/color_control_revAnimated_v122df83246f8fc5cc1228e2ce388aba90e3.jpeg)

### color canny demo (dreamshaper_8)
![color_canny_demo](assets/page/color_control_dreamshaper_843c96234dd48789470e55dfeb824d971.jpeg)
![color_canny_demo](assets/page/color_control_dreamshaper_8706c192858d6da783ae0f4b1c3b180f9.jpeg)
![color_canny_demo](assets/page/color_control_dreamshaper_831bc42c2295ac137422437bea83dcb8b.jpeg)


## How to Use
### 1. colorful canny annotator
    from annotator import colorful_canny

    image = cv2.imread(jpg_path)
    image, _ = colorful_canny(image, short_len=512, colorful=True)
    control_image = Image.fromarray(image)
       
### 2. diffusers pipeline
    checkpoint = "lllyasviel/control_v11p_sd15_canny"
    basemodel_path = "runwayml/stable-diffusion-v1-5"
    
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        basemodel_path, controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
	    base_model, torch_dtype=torch.float16, 
        controlnet=[
		    controlnet_pose,
	    ],
    ).to("cuda")

    image = pipe("a cat walking in the snow", num_inference_steps=20, generator=torch.manual_seed(0), image=control_image).images[0]

    image.save('images/image_out.png')


## Disclaimer

This project strives to positively impact the domain of AI-driven image generation. Users are granted the freedom to create images using this tool, but they are expected to comply with local laws and utilize it in a responsible manner. **The developers do not assume any responsibility for potential misuse by users.**

## Thanks
This implementation is inspired by [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) and [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)

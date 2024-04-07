import streamlit as st
from PIL import Image
import io
from chatgptresponse import preprocessedresponse
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from image import grid_image_generator
import torch
import pickle

# Function to convert and download an image as JPEG
def download_image_as_jpeg(img):
    img_jpeg = img.convert('RGB')
    img_byte_arr = io.BytesIO()
    img_jpeg.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

# Function to convert and download an image as PNG
def download_image_as_png(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# Main function for the Recipe Generator application
def app():
    # Initialize session state for button_pressed if it's not already set
    if 'button_pressed' not in st.session_state:
        st.session_state.button_pressed = False

    # Set the background color and text color for the page
    st.markdown(
        """
        <style>
        body {
            background-color: #F0F3F4;
            color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Add a title and subtitle
    st.title("Welcome to Recipe Generator!")
    st.subheader("Let's Cook Up Some Delicious Ideas")

    # Add an image related to cooking or food
    st.image("static/images/cooking.jpg", use_column_width=True)

    # Explain what the app does
    st.write(
        "Are you looking for culinary inspiration? Recipe Generator is here to help you discover "
        "mouthwatering recipes tailored to your tastes and preferences."
    )

    st.write(
        "With Recipe Generator, you can:"
    )
    st.markdown(
        "- 🍽️ **Generate Recipes**: Simply tell us your preferred cuisine, ingredients, and dietary restrictions, "
        "and we'll generate unique recipes for you."
    )
    st.markdown(
        "- 📝 **Generate Images**: Simply get the images along with the instructions for visual understanding of the recipe."
    )
    st.markdown(
        "- 🧾 **Save Image**: Save the Generated Image for future use."
    )

    # Invite users to get started
    st.write("Are you ready to embark on a culinary adventure?")
    st.write("Get Started by clicking on Generator!!")

    # Button to trigger recipe generation
    button_pressed = st.button("Try Now!!")

    if button_pressed:
        st.session_state.button_pressed = True

    if st.session_state.get('button_pressed', False):
        st.title("Recipe Generator")
        input_text = st.text_input("Enter the prompt: ", placeholder="How to make Lasagna?")
        
        if st.button('Generate Recipe'):
            with st.spinner('Processing...'):
                torch_device = "cuda" if torch.cuda.is_available() else "cpu"
                # print(torch_device)
                imageslist = []
                # 1. Load the autoencoder model which will be used to decode the latents into image space. 
                vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

                # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

                # 3. The UNet model for generating the latents.
                unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")



                scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
                vae = vae.to(torch_device)
                text_encoder = text_encoder.to(torch_device)
                unet = unet.to(torch_device)
                #prompt = ["a photograph of an astronaut riding a horse"]
                #prompt = ["a person chopping vegetables"]
                prompt = preprocessedresponse(input_text)
                print(prompt)

                with open("newprompt", "wb") as f1:
                    pickle.dump(prompt, f1)
                for i in range(0, len(prompt)):
                    height = 512                        # default height of Stable Diffusion
                    width = 512                         # default width of Stable Diffusion

                    num_inference_steps = 100             # Number of denoising steps

                    guidance_scale = 7.5                # Scale for classifier-free guidance

                    generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise

                    batch_size = 1
                    text_input = tokenizer(prompt[i], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

                    with torch.no_grad():
                        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
                    max_length = text_input.input_ids.shape[-1]
                    uncond_input = tokenizer(
                        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                    )
                    with torch.no_grad():
                        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   

                    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                    latents = torch.randn(
                        (batch_size, unet.in_channels, height // 8, width // 8),
                        generator=generator,
                    )
                    latents = latents.to(torch_device)
                    print(latents.shape)

                    scheduler.set_timesteps(num_inference_steps)

                    latents = latents * scheduler.init_noise_sigma
                    from tqdm.auto import tqdm
                    from torch import autocast

                    for t in tqdm(scheduler.timesteps):
                        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                        latent_model_input = torch.cat([latents] * 2)

                        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        with torch.no_grad():
                            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                        # perform guidance
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = scheduler.step(noise_pred, t, latents).prev_sample
                        # scale and decode the image latents with vae
                    latents = 1 / 0.18215 * latents

                    with torch.no_grad():
                        image = vae.decode(latents).sample
                    name = str(i+1) + '.png' 
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                    images = (image * 255).round().astype("uint8")
                    pil_images = [Image.fromarray(image) for image in images]
                    pil_images[0].save(name)
                    imageslist.append(name)

                grid_image_generator(imageslist)

            st.success('Planning Generated Successfully!!')
            try:
                # Assuming you have an image named "Mix Veg.png" in the "Images/FineTuned" folder
                image = Image.open("image_grid.png")
                st.image(image, caption="Generated Image for the Recipe", use_column_width=True)
                
                jpeg_image_data = download_image_as_jpeg(image)
                png_image_data = download_image_as_png(image)

                col1, col2, col3, col4  = st.columns(4)

                with col1:
                    st.download_button(label="Download as JPEG", data=jpeg_image_data, file_name="recipe_image.jpg", mime="image/jpeg")

                with col2:
                    st.download_button(label="Download as PNG", data=png_image_data, file_name="recipe_image.png", mime="image/png")
            except Exception as e:
                st.error(f"Failed to load and display image: {e}")
import streamlit as st
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
# pipe = pipe.to("cuda")  # 如果使用 GPU
pipe.safety_checker = None

st.title("Image Generation App")
prompt = st.text_input("Enter your prompt:")
if st.button("Generate"):
    if prompt:
        with st.spinner("Generating..."):
            images = pipe(prompt=prompt).images
            st.image(images[0], caption="Generated Image", use_column_width=True)
    else:
        st.warning("Please enter a prompt!")

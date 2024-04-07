from PIL import Image, ImageDraw, ImageFont
import textwrap
import pickle

def grid_image_generator(imageslist):
    # Assume all images are the same size and text boxes have the same height
    image_width = 300
    image_height = 300
    text_box_height = 100

    # The total grid width is the sum of individual image widths
    grid_width = image_width * len(imageslist)
    # The total grid height is the sum of an individual image height and the text box height
    grid_height = image_height + text_box_height
    
    # Create a new image with a white background
    grid_image = Image.new("RGB", (grid_width, grid_height), color="white")

    # Load the prompts
    with open("newprompt", "rb") as f2:
        prompt = pickle.load(f2)

    # Use a readable font
    font = ImageFont.load_default()
    
    # Iterate over the images and prompts to compose the grid
    for i, image_path in enumerate(imageslist):
        # Load the image
        image = Image.open(image_path)
        image = image.resize((image_width, image_height))

        # Create a draw object
        draw = ImageDraw.Draw(grid_image)

        # Calculate the position of the image in the grid
        image_position = (i * image_width, text_box_height)

        # Paste the image into the grid
        grid_image.paste(image, image_position)

        # Prepare the text
        wrapped_text = textwrap.fill(prompt[i], width=40)  # Adjust the width as needed
        text_size = draw.textsize(wrapped_text, font=font)
        # Calculate text position (centered in text box area)
        text_x = (i * image_width) + ((image_width - text_size[0]) // 2)
        text_y = (text_box_height - text_size[1]) // 2

        # Draw the text on the grid image
        draw.text((text_x, text_y), wrapped_text, font=font, fill="black")

    # Save the composed image
    grid_image.save("image_grid.png")
    return grid_image

import gradio as gr

# Define the recipe steps as a list of dictionaries.
# Each dictionary contains the image path and the text for that step.
recipe_steps = [
    {
        "image": "https://img.freepik.com/free-photo/closeup-scarlet-macaw-from-side-view-scarlet-macaw-closeup-head_488145-3540.jpg?semt=ais_hybrid&w=740&q=80",
        "text": "### Step 1: Gather Ingredients 🥬 \n\nBefore you start, make sure you have all your ingredients ready. This includes fresh vegetables, a protein source (like chicken or tofu), and your favorite dressing.",
    },
    {
        "image": "https://img.freepik.com/free-photo/closeup-scarlet-macaw-from-side-view-scarlet-macaw-closeup-head_488145-3540.jpg?semt=ais_hybrid&w=740&q=80",
        "text": "### Step 2: Prepare the Vegetables 🔪 \n\nWash and chop your vegetables. A good dice or julienne can make a big difference in the texture and presentation of the salad. For example, finely chop the lettuce and thinly slice the cucumbers.",
    },
    {
        "image": "https://img.freepik.com/free-photo/closeup-scarlet-macaw-from-side-view-scarlet-macaw-closeup-head_488145-3540.jpg?semt=ais_hybrid&w=740&q=80",
        "text": "### Step 3: Combine and Dress 🥣 \n\nIn a large bowl, combine all the chopped vegetables. Add your protein source and generously drizzle with your preferred dressing. Toss everything together gently to ensure an even coating.",
    },
    {
        "image": "https://img.freepik.com/free-photo/closeup-scarlet-macaw-from-side-view-scarlet-macaw-closeup-head_488145-3540.jpg?semt=ais_hybrid&w=740&q=80",
        "text": "### Step 4: Serve and Enjoy! 🍽️ \n\nServe your delicious, freshly-made salad immediately. You can garnish with some croutons, seeds, or fresh herbs for an extra touch. Enjoy your healthy meal!",
    },
]


def display_recipe():
    """Generates the Gradio components for the recipe display."""

    # Iterate through the list of recipe steps and yield a Markdown and an Image component for each.
    # The 'gr.Markdown' component displays the text, and 'gr.Image' displays the image.
    for step in recipe_steps:
        with gr.Row():
            # Use 'scale=1' to make the image smaller
            gr.Image(
                step["image"], label=f"Step {recipe_steps.index(step) + 1}", scale=1)
            # Use 'scale=2' to make the text wider
            gr.Markdown(step["text"], 2)


with gr.Blocks(title="A Simple Gradio Recipe") as demo:
    gr.Markdown("# 🥗 Healthy Salad Recipe")
    display_recipe()

demo.launch()

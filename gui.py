import json
import gradio as gr
from AlgorithmClass import AlgorithmClass
from helper import MultimodalModel
from rag import Datahandle
algorithm = AlgorithmClass()
data_handler = Datahandle()

# ini ganti dengan rag beneran dan embedding custom # <-- disini bay

recipes = data_handler.get_recipes()
embeddings_all, embedings_ingredients = data_handler.get_embeddings_recipe(
    recipes)

# =============================================================


def render_steps(input_text=None):
    if not input_text or input_text.strip() == "":
        return "<p>Please upload an image or enter some text to see the recipe steps.</p>"
    result = algorithm.first_generate_recipe()
    html = '<div class="scrollable-steps">'
    for idx, step in enumerate(result['steps'], start=1):
        html += f"""
        <div class="step-item">
            <img src="{step['images'][0]}" width="100"><br>
            <div class="step-text">
                <b>Step {idx}:</b> {step['text']}
            </div>
        </div>
        """
    html += "</div>"
    return html


def upload_file(file):
    if file is None:
        return None
    return file.name  # kasih path biar bisa dipreview di gr.Image


def text_replace(file):
    model = MultimodalModel()
    return model.generate(file)


def generate_recipe(input_text):
    # bikin HTML dari text
    algorithm.reset()
    embedding_input = data_handler.get_embeddings_input(input_text)
    algorithm.mapping_input(input_text, embedding_input)
    algorithm.mapping_output(
        recipes, embeddings_all, embedings_ingredients)
    return render_steps(input_text), gr.update(visible=True), gr.update(visible=True)


def next_recommendation(input_text, rating):
    algorithm.rating_recipe(rating)
    return render_steps(input_text)


with gr.Blocks(
    css="""
    .scrollable-steps {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 8px;
    }
    .step-item {
        display: flex;
        flex-direction: row;  /* bikin image sama text horizontal */
        align-items: center;  /* biar rata tengah secara vertikal */
        margin-bottom: 12px;
        gap: 12px; /* jarak antar image & text */
    }

    .step-item img {
        max-width: 120px;  /* atur ukuran gambar */
        height: auto;
        border-radius: 8px;
    }

    .step-item .step-text {
        flex: 1; /* biar teks mengisi sisa space */
    }
    """
) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("Recipe Recommendation Here")
            uploader = gr.UploadButton("Upload a file", file_types=[
                                       "image"], file_count="single")
            preview = gr.Image(label="Uploaded Preview")
            ingredients = gr.Textbox(
                label="Type your ingredients here...", interactive=True)
            generate_btn = gr.Button("Generate Recipe")
            # hidden dulu

        with gr.Column(scale=1):
            steps_html = gr.HTML(render_steps(),
                                 label="Steps will appear here")  # awalnya kosong
            rating = gr.Slider(
                minimum=-5, maximum=5, step=1,
                label="Rating For Recommendation",
                value=3, interactive=True, visible=False
            )
            next_btn = gr.Button("Next Recommendation", visible=False)

    uploader.upload(upload_file, uploader, preview)
    uploader.upload(text_replace, uploader, ingredients)
    # klik generate -> munculkan rating + next_btn
    generate_btn.click(
        generate_recipe,
        inputs=[ingredients],
        outputs=[steps_html, rating, next_btn]
    )

    # klik next recommendation
    next_btn.click(
        next_recommendation,
        inputs=[ingredients, rating],
        outputs=[steps_html]
    )

demo.launch()

import gradio
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="about")

JS_FUNC = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""


########################################################################################################################


@app.get("/welcome")
def read_main():
    return {
        "message": "Welcome to the main FastAPI app, please go to the root of the url to get started!"
    }


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def generate_filler(file: gradio.utils.NamedString) -> [str, str]:
    """
    Generates a generic filler text while we wait for the
    agentic architecture to be shipped to production
    Args:
        file: a gradio.utils.NamedString instance which is received
              when an upload is made to the upload area of the UI.

    Returns:
        publishable: a static explanation of why the uploaded paper
                     may be publishable
        suggestion: a static list of generic advice for any research paper
    """
    if file is None:
        return "Please upload a paper to analyze.", "No suggestions available yet."

    filename = file.name if hasattr(file, "name") else "Unnamed file"
    file_size = file.size if hasattr(file, "size") else "Unknown size"

    publishable = (
        f"Analysis of {filename} (size: {file_size} bytes): \n\nBased on our initial assessment, your paper "
        f"shows promise but needs some revisions before it's ready for publication."
    )

    suggestion = (
        "Suggestions for improvement:\n\n1. Strengthen your literature review\n2. Clarify your methodology "
        "section\n3. Consider adding more data visualization\n4. Expand on the limitations of your study"
    )

    return publishable, suggestion


with gradio.Blocks(js=JS_FUNC) as ui:
    with gradio.Row():
        with gradio.Column(scale=1):
            input_file = gradio.File(label="Paper File")
            upload_button = gradio.UploadButton(
                label="Upload the paper here",
                interactive=True,
                file_count="single",
                file_types=[".pdf"],
            )

        with gradio.Column(scale=2):
            is_publishable = gradio.TextArea(
                lines=20,
                max_lines=100,
                placeholder="Is your paper publishable? Let's find out!",
                autoscroll=True,
                label="Publication Assessment",
            )

        with gradio.Column(scale=1):
            suggestions = gradio.TextArea(
                lines=20,
                max_lines=100,
                placeholder="Improve your paper using custom insights",
                autoscroll=True,
                label="Improvement Suggestions",
            )

    @upload_button.upload(inputs=upload_button, outputs=input_file)
    def update_upload_area(file: gradio.utils.NamedString) -> gradio.utils.NamedString:
        return file

    @input_file.change(inputs=input_file, outputs=[is_publishable, suggestions])
    def analyse_paper(file: gradio.utils.NamedString) -> [str, str]:
        return generate_filler(file)


app = gradio.mount_gradio_app(app, ui, path="/gradio")

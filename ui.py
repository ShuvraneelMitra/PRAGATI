import gradio
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="about")


@app.get("/welcome")
def read_main():
    return {"message": "Welcome to the main FastAPI app, please go to the root of the url to get started!"}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def analyze_paper(file):
    if file is None:
        return "Please upload a paper to analyze.", "No suggestions available yet."

    filename = file.name if hasattr(file, "name") else "Unnamed file"
    file_size = file.size if hasattr(file, "size") else "Unknown size"

    publishable = f"Analysis of {filename} (size: {file_size} bytes):\n\nBased on our initial assessment, your paper shows promise but needs some revisions before it's ready for publication."

    suggestion = ("Suggestions for improvement:\n\n1. Strengthen your literature review\n2. Clarify your methodology "
                  "section\n3. Consider adding more data visualization\n4. Expand on the limitations of your study")

    return publishable, suggestion


js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

with gradio.Blocks(js=js_func) as ui:
    with gradio.Row():
        with gradio.Column(scale=1):
            input_area = gradio.File(label="Paper File")
            upload_button = gradio.UploadButton(
                label='Upload the paper here',
                interactive=True,
                file_count="single"
            )

        with gradio.Column(scale=2):
            is_publishable = gradio.TextArea(
                lines=20,
                max_lines=100,
                placeholder="Is your paper publishable? Let's find out!",
                autoscroll=True,
                label="Publication Assessment"
            )

        with gradio.Column(scale=1):
            suggestions = gradio.TextArea(
                lines=20,
                max_lines=100,
                placeholder="Improve your paper using custom insights",
                autoscroll=True,
                label="Improvement Suggestions"
            )

    upload_button.upload(
        analyze_paper,
        inputs=[upload_button],
        outputs=[is_publishable, suggestions]
    )

app = gradio.mount_gradio_app(app, ui, path="/gradio")

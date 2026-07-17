"""Single-app UI: upload a video, click the crossing line, enter the
Nemotron API key, process -> results table, then semantic search over
everything logged so far (text or photo query)."""

import cv2
import gradio as gr

from vehicle_clip_search import config, clip_embedder, pipeline, vector_store

DISPLAY_W, DISPLAY_H = 720, 405
RESULT_HEADERS = ["ID", "Brand", "Color", "Plate", "Direction", "Time", "Description"]
SEARCH_HEADERS = ["ID", "Brand", "Color", "Plate", "Direction", "Time", "Similarity"]


def load_frame(video_path):
    if not video_path:
        return None, None, []
    frame = pipeline.get_first_frame(video_path)
    h, w = frame.shape[:2]
    state = {"raw_frame": frame, "scale_x": DISPLAY_W / w, "scale_y": DISPLAY_H / h}
    return cv2.resize(frame, (DISPLAY_W, DISPLAY_H)), state, []


def pick_point(frame_state, points, evt: gr.SelectData):
    if frame_state is None:
        return None, points
    if len(points) >= 2:
        points = []

    x, y = evt.index
    sx, sy = frame_state["scale_x"], frame_state["scale_y"]
    points = points + [(x / sx, y / sy)]

    display = cv2.resize(frame_state["raw_frame"], (DISPLAY_W, DISPLAY_H)).copy()
    for px, py in points:
        cv2.circle(display, (int(px * sx), int(py * sy)), 6, (255, 0, 0), -1)
    if len(points) == 2:
        p1 = (int(points[0][0] * sx), int(points[0][1] * sy))
        p2 = (int(points[1][0] * sx), int(points[1][1] * sy))
        cv2.line(display, p1, p2, (255, 0, 0), 2)
    return display, points


def reset_line(frame_state):
    if frame_state is None:
        return None, []
    return cv2.resize(frame_state["raw_frame"], (DISPLAY_W, DISPLAY_H)), []


def run_pipeline(video_path, points, api_key, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")
    if len(points) != 2:
        raise gr.Error("Click 2 points on the frame to set the crossing line.")
    if not api_key:
        raise gr.Error("Enter your Nemotron API key.")

    annotated_path, records = pipeline.process_video(
        video_path, points[0], points[1], api_key,
        progress_cb=lambda p: progress(p, desc="Processing video"),
    )
    rows = [[r["track_id"], r["brand"], r["color"], r["plate"], r["direction"], r["time"], r["description"]]
            for r in records]
    return annotated_path, rows


def run_search(text_query, image_path, photo_type):
    if image_path:
        vector = clip_embedder.embed_image(image_path)
        hits = vector_store.search_plates(vector) if photo_type == "Plate" else vector_store.search_cars(vector)
    elif text_query and text_query.strip():
        hits = vector_store.search_cars(clip_embedder.embed_text(text_query.strip()))
    else:
        return [], [], []

    gallery = [
        (h["entity"]["car_image"],
         f"#{i + 1}  {h['entity']['color']} {h['entity']['brand']} | {h['entity']['plate']} | sim {round(1 - h['distance'], 3)}")
        for i, h in enumerate(hits)
    ]
    rows = [[h["entity"]["track_id"], h["entity"]["brand"], h["entity"]["color"], h["entity"]["plate"],
             h["entity"]["direction"], h["entity"]["time"], round(1 - h["distance"], 3)] for h in hits]
    return gallery, rows, hits


def show_vehicle_details(hits, evt: gr.SelectData):
    if not hits or evt.index >= len(hits):
        return ""
    e = hits[evt.index]["entity"]
    return (
        f"### Vehicle #{evt.index + 1} — ID {e['track_id']}\n"
        f"- **Brand:** {e['brand']}\n"
        f"- **Color:** {e['color']}\n"
        f"- **Plate:** {e['plate']}\n"
        f"- **Direction:** {e['direction']}\n"
        f"- **Time:** {e['time']}\n"
        f"- **Description:** {e['description']}"
    )


with gr.Blocks(title="Tunisian Vehicle Search Using VLMs and CLIP") as demo:
    gr.HTML(
        '<div style="text-align:center;">'
        '<h1 style="color:#e67e22;font-weight:800;font-size:250%;margin-bottom:0;">'
        'Tunisian Vehicle Search Using VLMs and CLIP</h1>'
        '<p style="color:#888;margin-top:4px;">Developed by Wassim Hfaiedh</p>'
        '</div>'
    )

    frame_state = gr.State(None)
    line_points = gr.State([])

    with gr.Tab("Process Video"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload video")
                api_key_input = gr.Textbox(
                    label="Nemotron API key", type="password",
                    value=config.DEFAULT_NVIDIA_API_KEY,
                )
                with gr.Row():
                    reset_btn = gr.Button("Reset line")
                    process_btn = gr.Button("Process", variant="primary")
            with gr.Column():
                frame_display = gr.Image(label="Click 2 points to set the crossing line", interactive=False)

        annotated_video = gr.Video(label="Annotated result")
        results_table = gr.Dataframe(headers=RESULT_HEADERS, label="Detected vehicles")

        video_input.change(load_frame, [video_input], [frame_display, frame_state, line_points])
        frame_display.select(pick_point, [frame_state, line_points], [frame_display, line_points])
        reset_btn.click(reset_line, [frame_state], [frame_display, line_points])
        process_btn.click(run_pipeline, [video_input, line_points, api_key_input],
                           [annotated_video, results_table])

    with gr.Tab("Semantic Search"):
        with gr.Row():
            text_query = gr.Textbox(label="Text query", placeholder="e.g. silver peugeot")
            image_query = gr.Image(label="Or upload a photo", type="filepath")
            photo_type = gr.Radio(["Car", "Plate"], value="Car", label="Uploaded photo is a")
        with gr.Row():
            with gr.Column(scale=2):
                search_gallery = gr.Gallery(label="Matches", columns=4, height=500)
            with gr.Column(scale=1):
                vehicle_details = gr.Markdown(label="Vehicle details", value="Click a result to see its details.")
        search_table = gr.Dataframe(headers=SEARCH_HEADERS, label="Match details")
        search_hits = gr.State([])

        search_inputs = [text_query, image_query, photo_type]
        search_outputs = [search_gallery, search_table, search_hits]
        text_query.change(run_search, search_inputs, search_outputs, show_progress=False)
        image_query.upload(run_search, search_inputs, search_outputs, show_progress=False)
        image_query.clear(run_search, search_inputs, search_outputs, show_progress=False)
        photo_type.change(run_search, search_inputs, search_outputs, show_progress=False)
        search_gallery.select(show_vehicle_details, [search_hits], [vehicle_details])


if __name__ == "__main__":
    demo.launch()

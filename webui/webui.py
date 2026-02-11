import tempfile

import gradio as gr
import numpy as np
import pandas as pd
import yaml

from data.data_utils import adaptive_transfer_name
from webui.pipeline import ImmuBPIPipeline


with open("./webui/config.yaml") as f:
    name_pipe_dict = yaml.safe_load(f)

default_key = next(iter(name_pipe_dict.keys()))

pipe = ImmuBPIPipeline.from_pretrained(**name_pipe_dict[default_key])
pipe.to("cuda")
curr = default_key

hla2seq_dict = np.load("./dataset/const/hla_type2seq.npy", allow_pickle=True).item()
hla_list = list(hla2seq_dict.keys())


def read_table_file(file_obj: tempfile._TemporaryFileWrapper) -> pd.DataFrame:
    path = file_obj.name
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx"):
        df = pd.read_xlsc(path)
    else:
        raise TypeError("Unsupported file format(we only support .csv and .xlsx)")
    return df


def _predict_single(epitope: str, HLA: str):
    if len(HLA) < 34:
        HLA = adaptive_transfer_name(HLA)
        HLA = hla2seq_dict[HLA]
    pos_score = pipe(epitope, HLA)[0].item()
    return {"positive": pos_score, "negative": 1 - pos_score}


def _predict_table(table: pd.DataFrame):
    hla_list = []
    if "HLA" in table:
        hla_list = list(table["HLA"])
    elif "hla" in table:
        hla_list = list(table["hla"])
    elif "Hla" in table:
        hla_list = list(table["Hla"])

    hla_list = [hla2seq_dict[adaptive_transfer_name(hla)] if len(hla) < 34 else hla for hla in hla_list]

    epitope_list = []

    if "peptide" in table:
        epitope_list = list(table["peptide"])
    elif "Peptide" in table:
        epitope_list = list(table["Peptide"])
    elif "epitope" in table:
        epitope_list = list(table["epitope"])
    elif "Epitope" in table:
        epitope_list = list(table["Epitope"])

    batch_size = 64
    n = len(hla_list)
    i = 0
    score_list = []
    while i < n:
        hla_batch = hla_list[i : i + batch_size]
        epitope_batch = epitope_list[i : i + batch_size]
        score_batch = pipe(epitope_batch, hla_batch)
        score_list += score_batch.tolist()

        i += batch_size

    table["score"] = score_list

    return table


def predict(
    model_name: str,
    predict_from: str,
    epitope: str,
    HLA: str,
    table: pd.DataFrame,
):
    global curr
    global pipe
    if model_name != curr:
        pipe = ImmuBPIPipeline.from_pretrained(**name_pipe_dict[model_name])
        pipe.to("cuda")
        curr = model_name

    if predict_from == "single pair":
        return _predict_single(epitope, HLA), gr.update()
    elif predict_from == "pairs in table":
        return gr.update(), _predict_table(table)


with gr.Blocks(title="ImmuBPI") as demo:
    with gr.Row():
        model_manager = gr.Dropdown(
            choices=list(name_pipe_dict.keys()), value=default_key, label="model", scale=1, min_width=0
        )
        for i in range(4):
            gr.HTML()

    with gr.Row():
        epitope = gr.Textbox(label="peptide", scale=2)
        HLA = gr.Dropdown(choices=hla_list, label="HLA", allow_custom_value=True, scale=2)
        with gr.Column():
            predict_btn = gr.Button("predict", variant="primary", scale=1)
            from_radio = gr.Radio(
                choices=["single pair", "pairs in table"], value="single pair", show_label=False, container=False
            )

    label_distribution = gr.Label(label="API prediction")

    file = gr.File(label=".csv or .xlsx file with column names peptide and HLA", height=100)
    table = gr.Dataframe(
        headers=["peptide", "HLA", "score"],
        datatype=["str", "str", "number"],
        interactive=True,
        wrap=True,
    )

    file.upload(fn=read_table_file, inputs=[file], outputs=[table])
    predict_btn.click(
        fn=predict, inputs=[model_manager, from_radio, epitope, HLA, table], outputs=[label_distribution, table]
    )

    with gr.Accordion("Input examples", open=False):
        gr.Markdown("### Single Pair Examples")
        gr.Examples(
            examples=[
                ["ImmuBPI_B", "VMLQAPLFT", "HLA-A0201"],
                ["ImmuBPI_I", "VMLQAPLFT", "HLA-A0201"],
                ["ImmuBPI_I", "NTQTSDTLSK", "HLA-A1101"],
            ],
            inputs=[
                model_manager,
                epitope,
                HLA,
            ],
        )

        gr.Markdown("### Table Examples(create input table by upload a file or manually input)")
        example_table = pd.DataFrame.from_dict(
            {"peptide": ["VMLQAPLFT", "VMLQAPLFT", "NTQTSDTLSK"], "hla": ["HLA-A0201", "HLA-A0101", "HLA-A0101"]}
        )
        gr.Examples(examples=[[example_table]], inputs=[table])

demo.launch()

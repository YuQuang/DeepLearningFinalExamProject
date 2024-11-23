from funasr import AutoModel

model = AutoModel(
    model="funasr/paraformer-zh",
    model_revision="v2.0.4",
    hub="hf",
    disable_update=True
)
res = model.export(quantize=False)
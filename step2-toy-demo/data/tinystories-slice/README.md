# TinyStories 切片

来源:https://huggingface.co/datasets/roneneldan/TinyStories,train split,按流式读取开头 309 篇故事。许可 CDLA-Sharing-1.0。

分句正则 `(?<=[.!?])\s+`;保留 2–40 个 token 的句子,共 4003 句,丢弃 59 句。

分词器:`Qwen/Qwen3.5-0.8B` 修订 `2fc06364715b967f1860aea9cf38778875588b17`,`tokenizer.json` SHA-256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`。

文件:`slice.json`(句子、紧凑 id、紧凑 id 到 Qwen id 的映射、每个 token 的文本、元数据),`SHA256SUMS`。

重建:`.venv/bin/python -m toy_demo.build_slice --target 4000`

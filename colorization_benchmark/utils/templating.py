from pathlib import Path


def pretty_print(text: str):
    return text.replace("_", " ").capitalize()


def image_html(src: Path, web_root: Path, width: int = 200):
    return '<img src="{{\'/' + str(src.relative_to(web_root)) + '\' | relative_url }}" width="200"/>'


def table_header(method_name: str, benchmark_type: str):
    return f"""---
title: '{pretty_print(method_name)}: {pretty_print(benchmark_type)}'
layout: default
tag: {method_name}
category: {benchmark_type}
---
# {pretty_print(benchmark_type)}
## {pretty_print(method_name)}

| Task | Image #1 |  Image #2 |  Image #3 | Reference |
| ----------- | ----------- | ----------- | ----------- | ----------- |\n"""

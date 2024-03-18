from pathlib import Path


def pretty_print(text: str):
    return text.replace("_", " ").capitalize()


def image_html(src: Path, web_root: Path, width: int = 200):
    return '<img src="{{\'/' + str(src.relative_to(web_root)) + '\' | relative_url }}" width="200"/>'


def table_header(method_name: str, benchmark_type: str, headers: list[str]):
    header = "| "
    header2 = "| "
    for name in headers:
        header += f" {name} |"
        header2 += f" ----- |"
    return f"""---
title: '{pretty_print(method_name)}: {pretty_print(benchmark_type)}'
layout: default
tag: {method_name}
category: {benchmark_type}
---
# {pretty_print(benchmark_type)}
## {pretty_print(method_name)}

{header}
{header2}
"""


def footer(method_name: str, benchmark_type: str):
    return '''
### Other categories:

{% for p in site.pages %}
{% if p.tag == "''' + method_name + '''" and p.url != page.url %}
- [{{ p.title }}]({{ p.url | relative_url }})
{% endif %}
{% endfor %}'''

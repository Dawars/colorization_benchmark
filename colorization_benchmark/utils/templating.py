import datetime
from pathlib import Path

from colorization_benchmark.model_wrappers.base_colorizer import BaseColorizer


def pretty_print(text: str):
    return text.replace("_", " ").capitalize()


def image_html(src: Path, web_root: Path, width: int = 200):
    return '<img src="{{\'/' + str(src.relative_to(web_root)) + '\' | relative_url }}" width="200"/>'


def table_header(method_name: str, benchmark_type: str, headers: list[str], colorizer: BaseColorizer):
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
last_modified_at: '{datetime.datetime.utcnow()}'
---
# {pretty_print(benchmark_type)}
## {pretty_print(method_name)}

{colorizer.get_description(benchmark_type)}

{header}
{header2}
"""


def footer(method_name: str, benchmark_type: str, colorizer: BaseColorizer):
    return '''
### Additional Information

- Last updated: {{ "''' + str(datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")) + '''" | date: site.minima.date_format }}
- Paper: [''' + colorizer.get_paper_link() + ''']()


### Other categories:

{% for p in site.pages %}
{% if p.tag == "''' + method_name + '''" and p.url != page.url %}
- [{{ p.title }}]({{ p.url | relative_url }})
{% endif %}
{% endfor %}
'''
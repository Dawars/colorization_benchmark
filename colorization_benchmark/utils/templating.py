import datetime
from pathlib import Path

import torch

from colorization_benchmark.model_wrappers.base_colorizer import BaseColorizer


def pretty_print(text: str):
    return text.replace("_", " ").capitalize()


def image_html(src: Path, web_root: Path, width: int = 200):
    return f'![]({{{{ "{str(src.relative_to(web_root))}" | relative_url }}}}){{: width="200px"}}'


def page_header(method_name: str, benchmark_type: str, colorizer: BaseColorizer):
    return f"""---
title: '{pretty_print(method_name)}: {pretty_print(benchmark_type)}'
layout: default
tag: {method_name}
category: {benchmark_type}
last_modified_at: "{datetime.datetime.utcnow()}"
---
# {pretty_print(benchmark_type)}
## {pretty_print(method_name)}

Paper: <{{{{ "{colorizer.get_paper_link()}" | uri_escape  }}}}>{{:target="_blank"}}

{colorizer.get_description(benchmark_type)}

"""


def table_header(headers: list[str]):
    header = "| "
    header2 = "| "
    for name in headers:
        header += f" {name} |"
        header2 += f" ----- |"
    return f"{header}\n{header2}\n"


def get_gpu_info():
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_device = torch.cuda.get_device_properties(0)
        gpu_info = (f"{gpu_device.name} {round(gpu_device.total_memory / 1024 ** 3)} GB, "
                    f"Compute Capability {gpu_device.major}.{gpu_device.minor}")
    else:
        gpu_info = "None"

    return gpu_info


def get_experiment_info():
    return f"""
- Last updated: {{{{ "{str(datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))}" | date: site.minima.date_format }}}}
- GPU info: {get_gpu_info()}
- CUDA version:  {torch.version.cuda}
- PyTorch version: {torch.__version__}

"""


def footer(method_name: str, benchmark_type: str, colorizer: BaseColorizer):
    return f"""
### Additional Information

{{% include_relative footer.md %}}

### Other Categories:

{{% for p in site.pages %}}
{{% if p.tag == "{method_name}" and p.url != page.url %}}
- [{{{{p.title}}}}]({{{{p.url | relative_url}}}})
{{% endif %}}
{{% endfor %}}
"""

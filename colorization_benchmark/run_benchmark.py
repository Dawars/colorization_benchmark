import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from colorization_benchmark.models.deep_remaster import DeepRemaster
from colorization_benchmark.utils import chromaticity

benchmark_pairs_single = {
    "recolor_source": [[  # recolor target
        {"source": "fortepan_183722.jpg", "references": ["fortepan_183722.jpg"]},
        {"source": "fortepan_250610.jpg", "references": ["fortepan_250610.jpg"]},
        {"source": "fortepan_183723.jpg", "references": ["fortepan_183723.jpg"]},
        {"source": "fortepan_251236.jpg", "references": ["fortepan_251236.jpg"]}
    ]],
    "full_correspondence": [[  # full correspondence
        {"source": "fortepan_201867.jpg", "references": ["fortepan_183723.jpg"]},
        {"source": "fortepan_229825.jpg", "references": ["fortepan_183723.jpg"]},
        {"source": "fortepan_102400.jpg", "references": ["fortepan_183723.jpg"]},
    ], [  # linebreak
        {"source": "fortepan_201867.jpg", "references": ["fortepan_251236.jpg"]},
        {"source": "fortepan_229825.jpg", "references": ["fortepan_251236.jpg"]},
        {"source": "fortepan_102400.jpg", "references": ["fortepan_251236.jpg"]},
    ]],
    "partial_reference": [[  # partial reference
        {"source": "fortepan_201867.jpg", "references": ["fortepan_183722.jpg"]},
        {"source": "fortepan_229825.jpg", "references": ["fortepan_183722.jpg"]},
        {"source": "fortepan_102400.jpg", "references": ["fortepan_183722.jpg"]},
    ], [
        {"source": "fortepan_201867.jpg", "references": ["fortepan_250610.jpg"]},
        {"source": "fortepan_229825.jpg", "references": ["fortepan_250610.jpg"]},
        {"source": "fortepan_102400.jpg", "references": ["fortepan_250610.jpg"]},
    ]],
    "partial_source": [[  # partial source
        {"source": "fortepan_18476.jpg", "references": ["fortepan_183723.jpg"]},
        {"source": "fortepan_79821.jpg", "references": ["fortepan_183723.jpg"]},
        {"source": "fortepan_67270.jpg", "references": ["fortepan_183723.jpg"]},
    ], [

        {"source": "fortepan_18476.jpg", "references": ["fortepan_251236.jpg"]},
        {"source": "fortepan_79821.jpg", "references": ["fortepan_251236.jpg"]},
        {"source": "fortepan_67270.jpg", "references": ["fortepan_251236.jpg"]},
    ]],
    "semantic_correspondence_strong": [[  # semantic correspondence
        {"source": "fortepan_251148.jpg", "references": ["fortepan_183723.jpg"]},
        {"source": "fortepan_97196.jpg", "references": ["fortepan_183723.jpg"]},
        {"source": "fortepan_97191.jpg", "references": ["fortepan_183723.jpg"]},
    ], [

        {"source": "fortepan_251148.jpg", "references": ["fortepan_251236.jpg"]},
        {"source": "fortepan_97196.jpg", "references": ["fortepan_251236.jpg"]},
        {"source": "fortepan_97191.jpg", "references": ["fortepan_251236.jpg"]},
    ]],
    "semantic_correspondence_weak": [[  # semantic correspondence
        {"source": "fortepan_148611.jpg", "references": ["fortepan_112161.jpg"]},
        {"source": "fortepan_84203.jpg", "references": ["fortepan_112161.jpg"]},
    ], [

        {"source": "fortepan_148611.jpg", "references": ["fortepan_115002.jpg"]},
        {"source": "fortepan_84203.jpg", "references": ["fortepan_115002.jpg"]},
    ]],
    "distractors": [[  # distractors, needs learning (semantic, foliage, people)
        {"source": "fortepan_18098.jpg", "references": ["fortepan_251236.jpg"]},
        {"source": "fortepan_276876.jpg", "references": ["fortepan_251236.jpg"]},
        {"source": "fortepan_40115.jpg", "references": ["fortepan_251236.jpg"]},
    ]],
    # "contemporary": [  # contemporary
    #     # {"source": "fortepan_197819.jpg", "references": [".jpg"]},
    # ]
}

benchmark_pairs_multi = {
    "recolor_source": [[  # recolor target
        {"source": "fortepan_183722.jpg",
         "references": ["fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
        {"source": "fortepan_250610.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
        {"source": "fortepan_183723.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_251236.jpg"]},
        {"source": "fortepan_251236.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg"]}
    ]],
    "full_correspondence": [[  # full correspondence
        {"source": "fortepan_201867.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
        {"source": "fortepan_229825.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
        {"source": "fortepan_102400.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
    ]],
    "partial_source": [[  # partial source
        {"source": "fortepan_18476.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
        {"source": "fortepan_79821.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
        {"source": "fortepan_67270.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
    ]],
    "semantic_correspondence_strong": [[  # semantic correspondence
        {"source": "fortepan_251148.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
        {"source": "fortepan_97196.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
        {"source": "fortepan_97191.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
    ]],
    "semantic_correspondence_weak": [[  # semantic correspondence
        {"source": "fortepan_148611.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
        {"source": "fortepan_84203.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
    ]],
    "distractors": [[  # distractors, needs learning (semantic, foliage, people)
        {"source": "fortepan_18098.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
        {"source": "fortepan_276876.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
        {"source": "fortepan_40115.jpg",
         "references": ["fortepan_183722.jpg", "fortepan_250610.jpg", "fortepan_183723.jpg", "fortepan_251236.jpg"]},
    ]],
    # "contemporary": [  # contemporary
    #     # {"source": "fortepan_197819.jpg", "references": [".jpg"]},
    # ]
}


def run_benchmark(colorizer, image_dir: Path, output_dir: Path):
    benchmark_type = "single_reference"
    method_name = colorizer.method_name
    web_root = output_dir.parent
    table_md = (f"""---
title: '{method_name.capitalize()}: Single Reference'
layout: default
---
# {benchmark_type.replace("_", " ").capitalize()}
## {method_name.replace("_", " ").capitalize()}

|  Image 1 |  Image 2 |  Image 3 |
| ----------- | ----------- | ----------- |\n""")

    experiment_root = output_dir / benchmark_type / method_name  # save table here
    for task_name, tasks in tqdm(benchmark_pairs_single.items()):
        i = 0
        for task in tasks:
            # table_line = f"| {task_name} | "
            table_line = f"| "
            for row in task:
                task_dir = output_dir / benchmark_type / method_name / task_name / str(i)
                task_dir.mkdir(exist_ok=True, parents=True)
                source_name = image_dir / row["source"]
                references = [image_dir / name for name in row["references"]]

                i += 1
                save_path = task_dir / f"{source_name.with_suffix('').name}_color.png"
                color = colorizer.colorize(source_name, references)
                color.save(save_path)

                xy_coordinates = chromaticity.image_to_cie_xy(save_path)
                chromaticity.plot_xy_coordinates_with_color(xy_coordinates, str(task_dir / f"{source_name.with_suffix('').name}_chromaticity.png"))

                table_line += '<img src="{{\'/' + str(save_path.relative_to(web_root)) + '\' | relative_url }}" width="200" />  |'
            table_line += '<img src="{{\'/' + str(references[0].relative_to(web_root)) + '\' | relative_url }}" width="200" />  |'  # assume same reference in row
            table_md += table_line + "\n"
    (experiment_root / "index.md").write_text(table_md)

    # Multi reference
    benchmark_type = "multi_reference"
    method_name = colorizer.method_name
    experiment_root = output_dir / benchmark_type / method_name  # save table here

    table_md = (f"""---
title: '{method_name.replace("_", " ").capitalize()}: Multi Reference'
layout: default
---
# {benchmark_type.replace("_", " ").capitalize()}
## {method_name.replace("_", " ").capitalize()}

| Image 1 |  Image 2 |  Image 3 |
| ----------- | ----------- | ----------- |\n""")

    for task_name, tasks in tqdm(benchmark_pairs_multi.items()):
        i = 0
        for task in tasks:
            # table_line = f"| {task_name} | "
            table_line = f"| "
            for row in task:
                task_dir = output_dir / benchmark_type / method_name / task_name / str(i)
                task_dir.mkdir(exist_ok=True, parents=True)
                source_name = image_dir / row["source"]
                references = [image_dir / name for name in row["references"]]

                i += 1
                save_path = task_dir / f"{source_name.with_suffix('').name}_color.png"
                color = colorizer.colorize(source_name, references)
                color.save(save_path)

                xy_coordinates = chromaticity.image_to_cie_xy(save_path)
                chromaticity.plot_xy_coordinates_with_color(xy_coordinates, str(task_dir / f"{source_name.with_suffix('').name}_chromaticity.png"))

                table_line += '<img src="{{\'/' + str(save_path.relative_to(web_root)) + '\' | relative_url }}" width="200" />  |'
            table_line += '<img src="{{\'/' + str(references[0].relative_to(web_root)) + '\' | relative_url }}" width="200" />  |'  # assume same reference in row
            table_md += table_line + "\n"
    (experiment_root / "index.md").write_text(table_md)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recoloring')
    parser.add_argument('--image_dir', type=Path, required=True, help='Input image root')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output dir')
    opt = parser.parse_args()

    output_dir: Path = opt.output_dir
    image_dir: Path = opt.image_dir
    model_path = "/home/dawars/projects/colorization/siggraphasia2019_remastering/model/remasternet.pth.tar"
    colorizer = DeepRemaster(model_path, mindim=320)
    run_benchmark(colorizer, image_dir, output_dir)

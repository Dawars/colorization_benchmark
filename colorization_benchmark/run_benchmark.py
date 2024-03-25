import argparse
import shutil
from pathlib import Path

import torch
import lightning
from tqdm import tqdm

from colorization_benchmark.model_wrappers.base_colorizer import BaseColorizer
from colorization_benchmark.utils import chromaticity
from colorization_benchmark.utils import templating

JPG_QUALITY = 95

benchmark_pairs_unconditional = {
    "recolor_source": [[  # recolor target
        {"source": "fortepan_183722.jpg", "references": []},
        {"source": "fortepan_250610.jpg", "references": []},
        {"source": "fortepan_183723.jpg", "references": []},
        {"source": "fortepan_251236.jpg", "references": []},
        {"source": "fortepan_201867.jpg", "references": []},
        {"source": "fortepan_229825.jpg", "references": []},
        {"source": "fortepan_102400.jpg", "references": []},
        {"source": "fortepan_18476.jpg", "references": []},
        {"source": "fortepan_79821.jpg", "references": []},
        {"source": "fortepan_67270.jpg", "references": []},
        {"source": "fortepan_251148.jpg", "references": []},
        {"source": "fortepan_97196.jpg", "references": []},
        {"source": "fortepan_97191.jpg", "references": []},
        {"source": "fortepan_148611.jpg", "references": []},
        {"source": "fortepan_84203.jpg", "references": []},
        {"source": "fortepan_18098.jpg", "references": []},
        {"source": "fortepan_276876.jpg", "references": []},
        {"source": "fortepan_40115.jpg", "references": []},
        {"source": "fortepan_197819.jpg", "references": []},
    ]]
}

benchmark_pairs_single = {
    "recolor_source": [[  # recolor target
        {"source": "fortepan_183722.jpg", "references": ["fortepan_183722.jpg"]},
        {"source": "fortepan_250610.jpg", "references": ["fortepan_250610.jpg"]},
        {"source": "fortepan_183723.jpg", "references": ["fortepan_183723.jpg"]},
        {"source": "fortepan_251236.jpg", "references": ["fortepan_251236.jpg"]}
    ]],
    "inverted_reference": [[  # check if unnatural colors work
        {"source": "fortepan_183722.jpg", "references": ["fortepan_183722_inv.jpg"]},
        {"source": "fortepan_250610.jpg", "references": ["fortepan_250610_inv.jpg"]},
        {"source": "fortepan_183723.jpg", "references": ["fortepan_183723_inv.jpg"]},
        {"source": "fortepan_251236.jpg", "references": ["fortepan_251236_inv.jpg"]}
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
        {"source": "fortepan_84203.jpg", "references": ["fortepan_112161.jpg"]},
    ], [

        {"source": "fortepan_148611.jpg", "references": ["fortepan_115002.jpg"]},
        {"source": "fortepan_84203.jpg", "references": ["fortepan_115002.jpg"]},
        {"source": "fortepan_84203.jpg", "references": ["fortepan_115002.jpg"]},
    ]],
    "distractors": [[  # distractors, needs learning (semantic, foliage, people)
        {"source": "fortepan_18098.jpg", "references": ["fortepan_251236.jpg"]},
        {"source": "fortepan_276876.jpg", "references": ["fortepan_251236.jpg"]},
        {"source": "fortepan_40115.jpg", "references": ["fortepan_251236.jpg"]},
    ]],
    "random_noise": [[  # random noise

        {"source": "fortepan_18098.jpg", "references": ["noise.jpg"]},
        {"source": "fortepan_276876.jpg", "references": ["noise.jpg"]},
        {"source": "fortepan_40115.jpg", "references": ["noise.jpg"]},
    ], [  # distractors, needs learning (semantic, foliage, people)
        {"source": "fortepan_201867.jpg", "references": ["noise.jpg"]},
        {"source": "fortepan_229825.jpg", "references": ["noise.jpg"]},
        {"source": "fortepan_102400.jpg", "references": ["noise.jpg"]},
    ]],
    "gray": [[  # check if color is hallucinated
        {"source": "fortepan_18098.jpg", "references": ["fortepan_18098.jpg"]},
        {"source": "fortepan_276876.jpg", "references": ["fortepan_276876.jpg"]},
        {"source": "fortepan_40115.jpg", "references": ["fortepan_40115.jpg"]},
    ], [
        {"source": "fortepan_201867.jpg", "references": ["fortepan_201867.jpg"]},
        {"source": "fortepan_229825.jpg", "references": ["fortepan_229825.jpg"]},
        {"source": "fortepan_102400.jpg", "references": ["fortepan_102400.jpg"]},
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
        {"source": "fortepan_84203.jpg",  # placeholder
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


def unconditional_benchmark(colorizer: BaseColorizer, image_dir: Path, output_dir: Path, markdown_only: bool):
    web_root = output_dir.parent
    benchmark_type = "unconditional"
    method_name = colorizer.method_name
    print(benchmark_type, method_name)

    experiment_root = output_dir / benchmark_type / method_name  # save table here
    if not markdown_only and experiment_root.exists():  # delete only dirs with pictures
        [shutil.rmtree(item) for item in experiment_root.iterdir() if item.is_dir()]

    table_md = templating.page_header(method_name, benchmark_type, colorizer)

    variation = False
    if variation:
        table_md += templating.table_header(["Image #1", "Image #2", "Image #3", "Image #4", "Image #5"])
    else:
        table_md += templating.table_header(["Image #1"])
    rows = 0
    for task_name, tasks in tqdm(benchmark_pairs_unconditional.items()):
        print(task_name)
        image_id = 0
        for row in tasks:
            for task in row:
                table_line = f'| '
                for i in range(5 if variation else 1):
                    lightning.seed_everything(i * 100)

                    task_dir = output_dir / benchmark_type / method_name / task_name / str(image_id)
                    task_dir.mkdir(exist_ok=True, parents=True)
                    source_name = image_dir / task["source"]

                    image_id += 1
                    save_path_color = task_dir / f"{source_name.with_suffix('').name}_color.jpg"
                    save_path_attention = task_dir / f"{source_name.with_suffix('').name}_attention.jpg"
                    save_path_chromaticity = task_dir / f"{source_name.with_suffix('').name}_chromaticity.png"
                    if not markdown_only:
                        with torch.no_grad():
                            results = colorizer.colorize(source_name, None)
                        color = results["color"]
                        attention = results.get("attention")
                        if color:
                            color.save(save_path_color, quality=JPG_QUALITY)
                        if attention:
                            attention.save(save_path_attention, quality=JPG_QUALITY)

                        if colorizer.generate_chromaticity():
                            xy_coordinates = chromaticity.image_to_cie_xy(save_path_color)
                            chromaticity.plot_xy_coordinates_with_color(xy_coordinates, str(save_path_chromaticity))

                    table_line += f"{templating.image_html(save_path_color, web_root)}"
                    if save_path_chromaticity.exists():
                        table_line += f"{templating.image_html(save_path_chromaticity, web_root)}"
                    if save_path_attention.exists():
                        table_line += f"{templating.image_html(save_path_attention, web_root)}"
                    table_line += " | "
                table_md += table_line + "\n"

    table_md += templating.footer(method_name, benchmark_type, colorizer)
    if not markdown_only:
        (experiment_root / "footer.md").write_text(templating.get_experiment_info())
    (experiment_root / "index.md").write_text(table_md)


def single_reference_benchmark(colorizer: BaseColorizer, image_dir: Path, output_dir: Path, markdown_only: bool):
    web_root = output_dir.parent
    benchmark_type = "single_reference"
    method_name = colorizer.method_name
    print(benchmark_type, method_name)

    experiment_root = output_dir / benchmark_type / method_name  # save table here
    # if not markdown_only and experiment_root.exists():  # delete only dirs with pictures
    #     [shutil.rmtree(item) for item in experiment_root.iterdir() if item.is_dir()]

    table_md = templating.page_header(method_name, benchmark_type, colorizer)
    table_md += templating.table_header(["Task", "Image #1", "Image #2", "Image #3", "Image #4"])
    rows = 0
    for task_name, tasks in tqdm(benchmark_pairs_single.items()):
        print(task_name)
        image_id = 0
        for row in tasks:
            if rows == 2:
                table_md += templating.table_header(["Task", "Image #1", "Image #2", "Image #3", "Reference"])

            table_line = f'| {templating.pretty_print(task_name)} | '
            rows += 1
            for task in row:
                task_dir = output_dir / benchmark_type / method_name / task_name / str(image_id)
                task_dir.mkdir(exist_ok=True, parents=True)
                source_name = image_dir / task["source"]
                references = [image_dir / name for name in task["references"]]

                image_id += 1
                save_path_color = task_dir / f"{source_name.with_suffix('').name}_color.jpg"
                save_path_attention = task_dir / f"{source_name.with_suffix('').name}_attention.jpg"
                save_path_chromaticity = task_dir / f"{source_name.with_suffix('').name}_chromaticity.png"
                if not markdown_only:
                    lightning.seed_everything(100)
                    with torch.no_grad():
                        results = colorizer.colorize(source_name, references)
                    color = results.get("color")
                    attention = results.get("attention")
                    if color:
                        color.save(save_path_color, quality=JPG_QUALITY)
                    if attention:
                        attention.save(save_path_attention, quality=JPG_QUALITY)

                    if colorizer.generate_chromaticity():
                        xy_coordinates = chromaticity.image_to_cie_xy(save_path_color)
                        chromaticity.plot_xy_coordinates_with_color(xy_coordinates, str(save_path_chromaticity))

                table_line += f"{templating.image_html(save_path_color, web_root)}"
                if save_path_chromaticity.exists():
                    table_line += f"{templating.image_html(save_path_chromaticity, web_root)}"
                if save_path_attention.exists():
                    table_line += f"{templating.image_html(save_path_attention, web_root)}"
                table_line += " |"
            if rows > 2:  # don't print ref in first row
                reference_chromaticity_path = references[
                                                  0].parent / f"{references[0].with_suffix('').name}_chromaticity.png"
                table_line += (f"{templating.image_html(references[0], web_root)}"
                               f"{templating.image_html(reference_chromaticity_path, web_root)} |")
            table_md += table_line + "\n"
    table_md += templating.footer(method_name, benchmark_type, colorizer)
    if not markdown_only:
        (experiment_root / "footer.md").write_text(templating.get_experiment_info())
    (experiment_root / "index.md").write_text(table_md)


def multi_reference_benchmark(colorizer: BaseColorizer, image_dir: Path, output_dir: Path, markdown_only: bool):
    web_root = output_dir.parent
    benchmark_type = "multi_reference"
    method_name = colorizer.method_name
    print(benchmark_type, method_name)

    experiment_root = output_dir / benchmark_type / method_name  # save table here
    if not markdown_only and experiment_root.exists():  # delete only dirs with pictures
        [shutil.rmtree(item) for item in experiment_root.iterdir() if item.is_dir()]

    table_md = templating.page_header(method_name, benchmark_type, colorizer)
    table_md += templating.table_header(["Task", "Image #1", "Image #2", "Image #3", "Reference"])

    rows = 0
    for task_name, tasks in tqdm(benchmark_pairs_multi.items()):
        print(task_name)
        image_id = 0
        for row in tasks:
            if rows == 1:
                table_md += templating.table_header(["Task", "Image #1", "Image #2", "Image #3", "Reference"])

            table_line = f'| {templating.pretty_print(task_name)} | '
            rows += 1
            for task in row:
                lightning.seed_everything(100)

                task_dir = output_dir / benchmark_type / method_name / task_name / str(image_id)
                task_dir.mkdir(exist_ok=True, parents=True)
                source_name = image_dir / task["source"]
                references = [image_dir / name for name in task["references"]]

                image_id += 1
                save_path_color = task_dir / f"{source_name.with_suffix('').name}_color.jpg"
                save_path_chromaticity = task_dir / f"{source_name.with_suffix('').name}_chromaticity.png"
                save_path_attention = task_dir / f"{source_name.with_suffix('').name}_attention.jpg"
                if not markdown_only:
                    with torch.no_grad():
                        results = colorizer.colorize(source_name, references)
                    color = results.get("color")
                    attention = results.get("attention")
                    if color:
                        color.save(save_path_color, quality=JPG_QUALITY)
                    if attention:
                        attention.save(save_path_attention, quality=JPG_QUALITY)

                    if colorizer.generate_chromaticity():
                        xy_coordinates = chromaticity.image_to_cie_xy(save_path_color)
                        chromaticity.plot_xy_coordinates_with_color(xy_coordinates, str(save_path_chromaticity))

                table_line += f"{templating.image_html(save_path_color, web_root)}"
                if save_path_chromaticity.exists():
                    table_line += f"{templating.image_html(save_path_chromaticity, web_root)} |"
                if save_path_attention.exists():
                    table_line += f"{templating.image_html(save_path_attention, web_root)} |"
            if rows > 1:  # don't print ref in first row
                reference_chromaticity_path = references[
                                                  0].parent / f"{references[0].with_suffix('').name}_chromaticity.png"
                table_line += (f"{templating.image_html(references[0], web_root)}"
                               f"{templating.image_html(reference_chromaticity_path, web_root)} |")
            table_md += table_line + "\n"

    table_md += templating.footer(method_name, benchmark_type, colorizer)
    if not markdown_only:
        (experiment_root / "footer.md").write_text(templating.get_experiment_info())
    (experiment_root / "index.md").write_text(table_md)


def run_benchmark(colorizer: BaseColorizer, image_dir: Path, output_dir: Path, unconditional: bool,
                  single_reference: bool, multi_reference: bool, markdown_only=False):
    if unconditional:
        unconditional_benchmark(colorizer, image_dir, output_dir, markdown_only)
    if single_reference:
        single_reference_benchmark(colorizer, image_dir, output_dir, markdown_only)
    if multi_reference:
        multi_reference_benchmark(colorizer, image_dir, output_dir, markdown_only)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recoloring')
    parser.add_argument('--image_dir', type=Path, required=True, help='Input image root')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output dir')
    opt = parser.parse_args()

    output_dir: Path = opt.output_dir
    image_dir: Path = opt.image_dir

    markdown_only = False

    from colorization_benchmark.model_wrappers.deep_remaster import DeepRemaster
    model_path = Path("../third_party/deepremaster/model/remasternet.pth.tar")
    colorizer = DeepRemaster(model_path, mindim=320)
    run_benchmark(colorizer, image_dir, output_dir, False, True, True, markdown_only=markdown_only)

    from colorization_benchmark.model_wrappers.unicolor import UniColor
    model_path = Path("../third_party/unicolor/framework/checkpoints/unicolor_mscoco/mscoco_step259999")
    colorizer = UniColor(model_path)
    run_benchmark(colorizer, image_dir, output_dir, False, True, False, markdown_only=markdown_only)
    # run_benchmark(colorizer, image_dir, output_dir, True, False, False, markdown_only=markdown_only)

    from colorization_benchmark.model_wrappers.pdnla_net import PDLNANet
    model_path = Path("../third_party/pdnla_net/model_1_ema.pt")
    colorizer = PDLNANet(model_path)
    run_benchmark(colorizer, image_dir, output_dir, False, True, False, markdown_only=markdown_only)

    # from colorization_benchmark.model_wrappers.inst_colorization import InstColorization
    # model_path = Path("../third_party/inst_colorization").absolute()
    # colorizer = InstColorization(model_path)
    # run_benchmark(colorizer, image_dir, output_dir, True, False, False, markdown_only=markdown_only)

    from colorization_benchmark.model_wrappers.colorize_net import ColorizeNet
    model_path = Path("../third_party/colorize_net").absolute()
    colorizer = ColorizeNet(model_path)
    run_benchmark(colorizer, image_dir, output_dir, True, False, False, markdown_only=markdown_only)
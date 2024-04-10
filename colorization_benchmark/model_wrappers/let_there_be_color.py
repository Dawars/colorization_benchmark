"""
Let there be Color!: Automatic Colorization of Grayscale Images: https://github.com/satoshiiizuka/siggraph2016_colorization
Requirements: docker

Note: colorization takes 25min instead of 2min for some reason. And the results differ significantly from the original
web API
"""
import sys
from pathlib import Path
from typing import List

import docker
from PIL import Image
import numpy as np

from colorization_benchmark.model_wrappers.base_colorizer import BaseColorizer


class LetThereBeColor(BaseColorizer):

    def __init__(self, model_path: Path, **opts):
        super().__init__("let_there_be_color")
        self.model_path = model_path
        self.client = docker.from_env()
        self.client.images.pull("italomaia/torch:lua53")
        image_dir = str(opts["image_dir"])
        self.container = self.client.containers.run(
            image="nagadomi/torch7",
            tty=True,
            auto_remove=True, detach=True,
            command="bash",
            volumes={
                model_path: {
                    'bind': '/root/siggraph2016_colorization',
                    'mode': 'rw',
                },
                image_dir: {
                    'bind': image_dir,
                    'mode': 'ro',
                },
            },
        )

    def __del__(self):
        exit_code, stream = self.container.exec_run(cmd="rm out.png", workdir="/root/siggraph2016_colorization/",
                                                    detach=False, stream=False)
        self.container.stop()
        self.container.remove()

    def get_paper_link(self):
        return "http://iizuka.cs.tsukuba.ac.jp/projects/colorization/web/"

    def colorize(self, input_path: Path, reference_paths: List[Path]) -> dict[str, Image]:
        out_path = self.model_path / "out.png"
        command = f"""bash -c "time /root/torch/install/bin/th colorize.lua '{str(input_path)}' out.png" """
        # command = f"""bash -c "time /root/torch/install/bin/th colorize.lua ansel_colorado_1941.png out.png" """
        print(command)
        # self.container.start()
        exit_code, stream = self.container.exec_run(cmd=command, workdir="/root/siggraph2016_colorization/",
                                                    detach=False, stream=False, tty=True)
        print(exit_code, stream)
        color = Image.open(out_path)
        return {"color": color}

    def download_model(self):
        ckpt_path = (Path(__file__).parent.parent.parent /
                     "third_party/siggraph2016_colorization/colornet.t7")
        md5_sum = "c88fa2bb6dc9f942a492a7dc7009b966"
        if not ckpt_path.exists():
            ckpt_path.parent.mkdir(exist_ok=True, parents=True)
            import wget

            try:
                wget.download("http://iizuka.cs.tsukuba.ac.jp/data/colornet.t7",
                              out=str(ckpt_path.parent))
            except:
                print(
                    "Download manually from http://web.archive.org/web/20210702050941/http://iizuka.cs.tsukuba.ac.jp/data/colornet.t7"
                    f"and place to {ckpt_path}")
        import hashlib
        assert hashlib.md5(ckpt_path.read_bytes()).hexdigest() == md5_sum

class BaseColorizer:
    def __init__(self, method_name: str):
        self.method_name = method_name

    def get_description(self, benchmark_type: str):
        return ""

    def generate_chromaticity(self):
        return True

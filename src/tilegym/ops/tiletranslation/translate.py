import argparse
import re

from pathlib import Path

def pass_device_context(content):
    return content.replace(
        b"torch.cuda.device(",
        b"torch.get_device_module().device(",
    )

def pass_l2_cache_size(content):
    return content.replace(
        b"torch.cuda.get_device_properties(0).L2_cache_size",
        b"ts.get_l2_cache_size()",
    )

def pass_device_capability(content):
    return content.replace(
        b"torch.cuda.get_device_capability(",
        b"ts.get_device_capability(",
    )

def pass_translator_import(content):
    return b"from tilegym.ops.tiletranslation import translator as ts\n" + content

def pass_launch(content):
    return content.replace(b"ct.launch(", b"ts.launch(")

def pass_current_stream(content):
    return content.replace(
        b"torch.cuda.current_stream(",
        b"torch.accelerator.current_stream(",
    )
def pass_utils_import(content):
    return content.replace(
        b"from .utils import",
        b"from tilegym.ops.cutile.utils import",
    )
def pass_backend(content):
    return content.replace(
        b'backend="cutile"',
        b'backend="tiletranslation"',
    )
def pass_multi_processor_count(content):
    return re.sub(
        rb"torch\.cuda\.get_device_properties\([^()]*\)\.multi_processor_count",
        b"ts.multi_processor_count()",
        content,
    )
def pass_compiler_timeout(content):
    return re.sub(
        rb"ct\.compiler_timeout\([^()]*\)",
        b"ts.dummy()",
        content,
    )
def pass_exhaustive_search(content):
    return content.replace(
        b"ct.tune.exhaustive_search", b"ts.exhaustive_search"
    ).replace(
        b"ct.exhaustive_search", b"ts.exhaustive_search"
    )

def pass_exhaustive_search_import(content):
    return content.replace(
        b"from cuda.tile.tune import exhaustive_search",
        b"from .translator import exhaustive_search",
    )

def pass_function_rename(content):
    return re.sub(
        rb"(?m)^([ \t]*(?:async[ \t]+)?def[ \t]+\w+)_cutile(?=[ \t]*\()",
        rb"\1_tiletranslation",
        content,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--rename-cutile-functions", action="store_true")
    args = parser.parse_args()

    content = args.input_file.read_bytes()
    for transformation in (
        pass_launch,
        pass_current_stream,
        pass_utils_import,
        pass_translator_import,
        pass_backend,
        pass_multi_processor_count,
        pass_compiler_timeout,
        pass_exhaustive_search,
        pass_exhaustive_search_import,
	pass_device_capability,
	pass_l2_cache_size,
	pass_device_context,
    ):
        content = transformation(content)

    if args.rename_cutile_functions:
        content = pass_function_rename(content)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("wb") as output:
        output.write(content)

    print(f"Created: {args.output_file.resolve()}")

if __name__ == "__main__":
    main()

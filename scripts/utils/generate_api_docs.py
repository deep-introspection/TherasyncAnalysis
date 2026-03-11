"""
Generate API documentation from source code.

This script introspects Python modules to extract accurate API information
including class names, method signatures, parameters, and docstrings.
"""

import inspect
import importlib
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def get_class_methods(cls) -> List[Dict[str, Any]]:
    """Extract public methods from a class."""
    methods = []

    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        # Skip private methods (except __init__)
        if name.startswith("_") and name != "__init__":
            continue

        sig = inspect.signature(method)
        doc = inspect.getdoc(method) or "No documentation available."

        methods.append(
            {
                "name": name,
                "signature": str(sig),
                "params": list(sig.parameters.keys()),
                "docstring": doc,
                "return_annotation": sig.return_annotation,
            }
        )

    return methods


def format_method_doc(method: Dict[str, Any], class_name: str) -> str:
    """Format a method as markdown documentation."""
    name = method["name"]
    sig = method["signature"]
    doc = method["docstring"]

    # Build full signature
    full_sig = f"{name}{sig}"

    md = f"##### `{full_sig}`\n\n"

    # Add docstring (indented properly)
    if doc:
        # Clean up docstring formatting
        lines = doc.split("\n")
        cleaned_lines = []
        for line in lines:
            cleaned_lines.append(line)
        md += "\n".join(cleaned_lines) + "\n\n"

    return md


def document_class(module_name: str, class_name: str) -> str:
    """Generate documentation for a class."""
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        doc = inspect.getdoc(cls) or "No documentation available."
        methods = get_class_methods(cls)

        md = f"#### Class: `{class_name}`\n\n"
        md += f"**Module**: `{module_name}`\n\n"
        md += f"{doc}\n\n"

        # Constructor
        init_method = next((m for m in methods if m["name"] == "__init__"), None)
        if init_method:
            md += "**Constructor**:\n\n"
            md += f"```python\n{class_name}{init_method['signature']}\n```\n\n"

        # Public methods
        public_methods = [m for m in methods if m["name"] != "__init__"]
        if public_methods:
            md += "**Methods**:\n\n"
            for method in public_methods:
                md += format_method_doc(method, class_name)

        return md

    except Exception as e:
        return f"#### Class: `{class_name}`\n\nError generating documentation: {e}\n\n"


def main():
    """Generate API documentation."""

    print("🔍 Generating API documentation from source code...\n")

    # Core modules
    print("📦 Core Modules:")
    config_doc = document_class("src.core.config_loader", "ConfigLoader")
    print("  ✓ ConfigLoader")

    # BVP Preprocessing
    print("\n📦 BVP Preprocessing:")
    bvp_loader_doc = document_class("src.physio.preprocessing.bvp_loader", "BVPLoader")
    print("  ✓ BVPLoader")

    bvp_cleaner_doc = document_class(
        "src.physio.preprocessing.bvp_cleaner", "BVPCleaner"
    )
    print("  ✓ BVPCleaner")

    bvp_metrics_doc = document_class(
        "src.physio.preprocessing.bvp_metrics", "BVPMetricsExtractor"
    )
    print("  ✓ BVPMetricsExtractor")

    bvp_writer_doc = document_class(
        "src.physio.preprocessing.bvp_bids_writer", "BVPBIDSWriter"
    )
    print("  ✓ BVPBIDSWriter")

    # EDA Preprocessing
    print("\n📦 EDA Preprocessing:")
    eda_loader_doc = document_class("src.physio.preprocessing.eda_loader", "EDALoader")
    print("  ✓ EDALoader")

    eda_cleaner_doc = document_class(
        "src.physio.preprocessing.eda_cleaner", "EDACleaner"
    )
    print("  ✓ EDACleaner")

    eda_metrics_doc = document_class(
        "src.physio.preprocessing.eda_metrics", "EDAMetricsExtractor"
    )
    print("  ✓ EDAMetricsExtractor")

    eda_writer_doc = document_class(
        "src.physio.preprocessing.eda_bids_writer", "EDABIDSWriter"
    )
    print("  ✓ EDABIDSWriter")

    # HR Preprocessing
    print("\n📦 HR Preprocessing:")
    hr_loader_doc = document_class("src.physio.preprocessing.hr_loader", "HRLoader")
    print("  ✓ HRLoader")

    hr_cleaner_doc = document_class("src.physio.preprocessing.hr_cleaner", "HRCleaner")
    print("  ✓ HRCleaner")

    hr_metrics_doc = document_class(
        "src.physio.preprocessing.hr_metrics", "HRMetricsExtractor"
    )
    print("  ✓ HRMetricsExtractor")

    hr_writer_doc = document_class(
        "src.physio.preprocessing.hr_bids_writer", "HRBIDSWriter"
    )
    print("  ✓ HRBIDSWriter")

    # Save to file
    output_file = project_root / "docs" / "api_reference_generated.md"

    with open(output_file, "w") as f:
        f.write("# API Reference - TherasyncPipeline (Auto-Generated)\n\n")
        f.write("**Version**: 1.0.0 (Production Ready)\n")
        f.write("**Last Updated**: November 11, 2025\n\n")
        f.write("---\n\n")

        f.write("## Core Modules\n\n")
        f.write(config_doc)

        f.write("\n---\n\n## BVP Preprocessing\n\n")
        f.write(bvp_loader_doc)
        f.write(bvp_cleaner_doc)
        f.write(bvp_metrics_doc)
        f.write(bvp_writer_doc)

        f.write("\n---\n\n## EDA Preprocessing\n\n")
        f.write(eda_loader_doc)
        f.write(eda_cleaner_doc)
        f.write(eda_metrics_doc)
        f.write(eda_writer_doc)

        f.write("\n---\n\n## HR Preprocessing\n\n")
        f.write(hr_loader_doc)
        f.write(hr_cleaner_doc)
        f.write(hr_metrics_doc)
        f.write(hr_writer_doc)

    print(f"\n✅ Generated API documentation: {output_file}")
    print(f"📄 File size: {output_file.stat().st_size} bytes")


if __name__ == "__main__":
    main()

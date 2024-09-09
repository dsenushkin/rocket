import rocket
import os


def generate_autoclass_files(output_dir: str) -> None:
    """
    Generate Sphinx autoclass files for classes in
    rocket.core.__sphinx_classes__.

    This function creates individual Markdown files for each class in the
    rocket.core.__sphinx_classes__ list. Each file contains the Sphinx autoclass
    directive for the corresponding class.

    Args:
        output_dir (str): The directory where the generated files will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    for cls in rocket.core.__sphinx_classes__:
        class_name = cls.__name__
        file_name = f"{class_name.lower()}.md"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, "w") as f:
            f.write(f"# {class_name}\n\n")
            f.write(f".. autoclass:: rocket.core.{class_name}\n")
            f.write("   :members:\n")
            f.write("   :undoc-members:\n")
            f.write("   :show-inheritance:\n")

    print(f"Generated autoclass files in {output_dir}")


if __name__ == "__main__":
    output_directory = "docs/source/api"
    generate_autoclass_files(output_directory)

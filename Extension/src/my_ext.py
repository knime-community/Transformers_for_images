import knime.extension as knext


main_category = knext.category(
    path="/community/",
    level_id="vit_ft",
    name="Vision Transformers",
    description="Nodes for fine-tuning Vision Transformers",
    icon="icons/icon.png",
)

from nodes import vit

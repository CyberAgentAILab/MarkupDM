# Crello-Instruct Dataset

This directory contains the files for the **Crello-Instruct** dataset, an extension of the [Crello](https://huggingface.co/datasets/cyberagent/crello) dataset for instruction-guided graphic design completion.

## Files

- `crello_captions.json`: Contains detailed captions for each image element in the Crello dataset. These annotations cover both semantically meaningful content (e.g., photos, illustrations) and decorative elements (e.g., shapes, lines) to help models understand the complete visual context.
- `crello_instructions.json`: Contains the 125K triplets' instructions. Each entry provides a natural language instruction for transitioning from a partial design to a completed design.

## Key Structure

Each entry in the JSON files uses a unique identifier as a key, following the format:
`{split}_{index}_{id}_{element_index}`

- **split**: The dataset split (e.g., `train`, `val`, `test`).
- **index**: The numerical index of the design.
- **id**: The original template ID from the Crello dataset.
- **element_index**: The index of the target element. For instructions, this is the index of dropped element.

Example: `train_0_5cab562223c829c8214df7e0_0` means the first element of the first design in the training set.

Each entry follows a key-value pair where the key is a unique identifier (linking to the Crello dataset templates) and the value is the corresponding natural language description.

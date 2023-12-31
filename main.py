from typing import Tuple
import json
import os
import re
import tempfile

import openai
import streamlit as st
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def get_notebook_context(path: str, cell_type: str = "code") -> Tuple[str, int]:
    """
    Get notebook context and token length from path.

    :param path: File path

    :param cell_type: Cell type ('code' or 'markdown')
    :return: Tuple containing cleaned notebook context and token length
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            notebook_code = json.load(f)["cells"]
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return "", 0

    if cell_type not in ["code", "markdown"]:
        raise ValueError("Invalid cell_type. Must be 'code' or 'markdown'.")

    notebook_code = [
        cell["source"] for cell in notebook_code if cell["cell_type"] == cell_type
    ]
    notebook_code = "\n".join([''.join(cell) for cell in notebook_code])

    # Removing comments and URLs
    notebook_code = re.sub("#.*?\\n", "", notebook_code)
    notebook_code = re.sub(r"https?:\/\/.*[\r\n]*", "", notebook_code)

    token_length = len(notebook_code.split())
    return notebook_code, token_length


def summary(model_openai: str = "gpt-3.5-turbo-16k", notebook_code: str = None) -> str:
	"""
	Generate summary from notebook code
	:param model_openai: OpenAI model
	:param notebook_code: Notebook code
	:return: Summary
	"""
	response = openai.ChatCompletion.create(
		model=model_openai,
		messages=[
			{
				"role": "system",
				"content": "You are my jupyter notebook summarizer. You will summarize the code I given into 1. Business Statement 2. Table of contents 3. Conclusion & Executive Summary 4. Recommendations write this concise, clear and to the point. like senior data scientist would do. Return only the answer of summarized text, no need to return the question",
			},
			{"role": "user", "content": f"Here the code {notebook_code}"},
		],
	)
	summary = response["choices"][0]["message"]["content"]
	print(f"Length of summary: {len(summary.split())}")
	return summary


def calculate_bleu(reference_text: str, candidate_text: str) -> float:
	"""
	Calculate BLEU score between reference and candidate sentences.

	Args:
	reference (List[str]): List of words in the reference sentence.
	candidate (List[str]): List of words in the candidate sentence.

	Returns:
	float: BLEU score
	"""
	smoothie = SmoothingFunction().method4
	return sentence_bleu([reference_text], candidate_text, smoothing_function=smoothie)


def get_score(original, summary):
	"""
	Calculate rouge score and bleu score
	:param original: Original text
	:param summary: Summary text
	:return: Rouge score and bleu score
	"""
	# calculate rouge score
	scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
	scores = scorer.score(original, summary)
	rouge_score = scores["rouge1"]
	# calculate bleu score
	bleu_score = calculate_bleu(original.split(), summary.split())
	return rouge_score, bleu_score


def insert_and_save_summary(
		notebook_path: str, summary_text: str, new_suffix: str = "_summarized"
) -> None:
	"""
	Inserts the summary text after the first header and saves to a new notebook.

	:param notebook_path: The path of the original notebook
	:param summary_text: The summary text to insert
	:param new_suffix: The suffix to add to the new notebook filename
	"""
	# Load the notebook
	with open(notebook_path, "r", encoding="utf-8") as f:
		notebook = json.load(f)

	# Create a new markdown cell with the summary
	summary_cell = {"cell_type": "markdown", "metadata": {}, "source": [summary_text]}

	# Find the first header cell and insert summary cell after it
	for i, cell in enumerate(notebook["cells"]):
		if cell["cell_type"] == "markdown":
			if re.search(r"^#+ ", "".join(cell["source"])):
				notebook["cells"].insert(i + 1, summary_cell)
				break
	else:
		# If no header is found, append at the end
		notebook["cells"].append(summary_cell)

	# Save to a new file
	new_file_path = os.path.splitext(notebook_path)[0] + new_suffix + ".ipynb"
	with open(new_file_path, "w", encoding="utf-8") as f:
		json.dump(notebook, f, ensure_ascii=False, indent=4)


# Streamlit app
def main():
	st.image("img.jpg", use_column_width=True)
	st.title("Jupyter Notebook Summarizer")

	# Upload notebook file
	uploaded_file = st.file_uploader("Upload a Jupyter Notebook", type=["ipynb"])

	if uploaded_file is not None:
		# Create a temporary file
		with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as tfile:
			tfile.write(uploaded_file.read())

		# Get notebook content
		notebook_code = get_notebook_context(tfile.name)

		# Generate summary
		summary_text = summary(notebook_code=notebook_code)

		# Insert summary and save to new notebook
		summarized_file = tfile.name.replace(".ipynb", "_summarized.ipynb")
		insert_and_save_summary(tfile.name, summary_text)

		# Calculate scores
		rouge_score, bleu_score = get_score(notebook_code, summary_text)

		# Display scores
		st.write(
			f"Rouge Score: {rouge_score.fmeasure:.2f}, BLEU Score: {bleu_score:.2f}"
		)

		# Display summarized text
		st.markdown("### Summarized Text")
		st.text(summary_text)

		# Create a download button
		with open(summarized_file, "rb") as f:
			bytes_data = f.read()
		st.download_button(
			label="Download Summarized Notebook",
			data=bytes_data,
			file_name="notebook_summarized.ipynb",
			mime="application/x-ipynb+json",
		)

		# Cleanup temporary files
		os.remove(tfile.name)
		os.remove(summarized_file)


# Combine this with your existing functions:
# get_notebook_context, summary, calculate_bleu, get_score, insert_and_save_summary

if __name__ == "__main__":
	main()

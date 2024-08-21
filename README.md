# Evaluating Dialect Robustness of Large Language Models via Conversation Understanding
**Authors:** Dipankar Srirag and Nihar Ranjan Sahoo and Aditya Joshi

**DOI:** [10.48550/arXiv.2405.05688](https://doi.org/10.48550/arXiv.2405.05688)

## Abstract
With an evergrowing number of LLMs reporting superlative performance for English, their ability to perform equitably for different dialects of English ($\textit{i.e.}$, dialect robustness) needs to be ascertained. Specifically, we use English language (US English or Indian English) conversations between humans who play the word-guessing game of `taboo`. We formulate two evaluative tasks: target word prediction (TWP) ($\textit{i.e.}$, predict the masked target word in a conversation) and target word selection (TWS) ($\textit{i.e.}$, select the most likely masked target word in a conversation, from among a set of candidate words). Extending [`MD-3`]((https://doi.org/10.48550/arXiv.2305.11355)), an existing dialectic dataset of taboo-playing conversations, we introduce `MMD-3`, a target-word-masked version of `MD-3` with the `en-US` and `en-IN` subsets. We create two subsets: `en-MV` (where `en-US` is transformed to include dialectal information) and `en-TR` (where dialectal information is removed from `en-IN`). We evaluate one open-source (Llama3) and two closed-source (GPT-4/3.5) LLMs. LLMs perform significantly better for US English than Indian English for both TWP and TWS tasks, for all settings, exhibiting marginalisation against the Indian dialect of English. While GPT-based models perform the best, the comparatively smaller models work more equitably after fine-tuning. Our error analysis shows that the LLMs can understand the dialect better after fine-tuning using dialectal data. Our evaluation methodology exhibits a novel way to examine attributes of language models using pre-existing dialogue datasets.

## Keywords
- Large Language Models
- Dialect Robustness
- Conversation Understanding
- Word-Guessing Game

## BibTeX Citation
<tab><tab>
```bibtex
@misc{srirag2024evaluating,
      title={Evaluating Dialect Robustness of Language Models via Conversation Understanding}, 
      author={Dipankar Srirag and Nihar Ranjan Sahoo and Aditya Joshi},
      year={2024},
      eprint={2405.05688},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact
- [Dipankar Srirag](mailto:d.srirag@unsw.edu.au); [University of New South Wales](https://dipankarsrirag.github.io)
- [Nihar Ranjan Sahoo](mailto:nihar@cse.iitb.ac.in); [Indian Institute of Technology Bombay](https://sahoonihar.github.io)
- [Aditya Joshi](mailto:aditya.joshi@unsw.edu.au); [University of New South Wales](https://www.unsw.edu.au/staff/aditya-joshi)
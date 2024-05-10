# Evaluating Dialect Robustness of Large Language Models via Conversation Understanding
**Authors:** Dipankar Srirag and Aditya Joshi

**DOI:** [10.48550/arXiv.2405.05688](https://doi.org/10.48550/arXiv.2405.05688)

## Abstract
With an evergrowing number of LLMs reporting superlative performance for English, their ability to perform equitably for different dialects of English (i.e., dialect robustness) needs to be ascertained. Specifically, we use English language (US English or Indian English) conversations between humans who play the word-guessing game of "taboo". We formulate two evaluative tasks: target word prediction (`TWP`) (i.e.predict the masked target word in a conversation) and target word selection (`TWS`) (i.e., select the most likely masked target word in a conversation, from among a set of candidate words). Extending [`MD3`](https://doi.org/10.48550/arXiv.2305.11355), an existing dialectic dataset of taboo-playing conversations, we introduce `M-MD3`, a target-word-masked version of MD3 with the <span style="color:#3C6C76">USEng</span> and <span style="color:#F17D7D">IndEng</span> subsets. We add two subsets: <span style="color:#9A70ED">AITrans</span> (where dialectic information is removed from <span style="color:#F17D7D">IndEng</span>) and <span style="color:#458657">AIGen</span> (where LLMs are prompted to generate conversations). 
Our evaluation uses pre-trained and fine-tuned versions of two closed-source (`GPT-4/3.5`) and two open-source LLMs (`Mistral` and `Gemma`). LLMs perform significantly better for US English than Indian English for both TWP and TWS, for all settings. While GPT-based models perform the best, the comparatively smaller models work more equitably for short conversations (`<8 turns`). Our results on <span style="color:#458657">AIGen</span> and <span style="color:#9A70ED">AITrans</span> (the best and worst-performing subset) respectively show that LLMs may learn a dialect of their own based on the composition of the training data, and that dialect robustness is indeed a challenging task. Our evaluation methodology exhibits a novel way to examine attributes of language models using pre-existing dialogue datasets.

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
      author={Dipankar Srirag and Aditya Joshi},
      year={2024},
      eprint={2405.05688},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact
- [Dipankar Srirag](mailto:d.srirag@student.unsw.edu.au); [University of New South Wales](https://www.unsw.edu.au)
- [Aditya Joshi](mailto:aditya.joshi@unsw.edu.au); [University of New South Wales](https://www.unsw.edu.au/staff/aditya-joshi)
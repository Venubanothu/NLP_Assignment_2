# Project 2 — Statistical vs. Neural Sequence Labeling

Dataset: UD English EWT (17 Universal POS tags)  
Seed reported: 42

## Contents
- `task1_hmm.py`: Task 1 HMM class and token-accuracy helper.
- `task2_bilstm_crf_reference.py`: Task 2 source-code reference extracted from the working code notes. The executed Colab notebook is the authoritative version.
- `task3_hybrid_notebook_reference.ipynb`: offline-safe Task 3 notebook reference. The final executed Colab notebook includes the vocabulary-bundle loading fix described below.
- `task3_final_analysis_output.txt`: final confusion and qualitative output (if supplied).
- `../results.json`: saved summary of final evaluation metrics.

## Reported test results
| Model | Overall | Known | Unknown |
|---|---:|---:|---:|
| HMM counts | 91.92% | 93.21% | 75.98% |
| BiLSTM-CRF | 95.72% | 96.50% | 86.13% |
| Neural-emission HMM | 93.73% | 94.82% | 80.29% |

## Environment
Core packages: Python 3, NumPy, PyTorch, `pytorch-crf`. Dataset loader in the executed notebook uses the UD English EWT CoNLL-U files. The exact package versions should be copied from the Colab environment used for the final run.

## Reproducibility / important notes
1. Use the exact Task 2 word/character vocabularies and embedding matrix for Task 3 encoder transfer. In the final run this was saved as `task2_vocab_bundle.pt` and loaded before creating `model3`.
2. The Task 2 checkpoint is `bilstm_crf.pt`; the Task 3 softmax checkpoint is `neural_softmax.pt`. These checkpoints live in the user's Google Drive and are not included in this source zip.
3. Export the **final executed** Task 1 and Task 2 Colab notebooks and replace/add them here before submitting. The original executed notebooks were not available as files in this packaging workspace, so the Task 2 file here is a reference export rather than a verified notebook export.
4. The final Task 3 run replaced the original vocabulary-building cell with the saved Task 2 bundle. Do not rerun the old vocabulary builder after loading the bundle.
5. The Task 3 offline setup printed that no local GloVe file was found. The final run then reused the Task 2 embedding matrix. Verify the Task 2 embedding provenance in its execution log and describe it accurately in the report.
6. Use the final saved JSON test accuracy (Task 2 95.72%) consistently; earlier training output showed a slightly different number.

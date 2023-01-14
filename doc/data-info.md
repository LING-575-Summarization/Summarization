# Info on dataset and directory layout

## Corpora info (from sldies)

- AQUAINT: `/corpora/LDC/LDC02T31/`
- AQUAINT-2: `/corpora/LDC/LDC08T25/data/`

## Dropbox directory info (from slides)

- `Data/Documents/`:  links of documents to be summarized
- `Subdir`: training/, devtest/ and evaltest/
- `Subdir/*.xml`: a list of “topics”
- `[dev|eval]test/categories.txt`: a list of “categories”
- `Data/models/`:  summaries written by humans
- `Data/peers/`: summaries created by share task participants 
- `Data/scores/`: human evaluation scores (including Pyramid)
- `Data/TAC_scores/`: ROUGE scores
- `code/ROUGE/`: code for calculating ROUGE
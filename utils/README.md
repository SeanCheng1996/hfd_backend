## `merge_datasets_hfd.py`

Code to run for merging the main dataset, `HFD MVA History_modified.xlsx`, with supplementary datasets `HFD Stations.xlsx` and `MVA Costs - 17-20.xlsx`

## `notes_utils.py`

Module containing helper functions for performing analysis on Notes, including (but not limited to)
   > * `do_nlp_process`: preprocessing notes
   > * `get_namedEntity_list`: generating named entity list
   > * `get_Nouns_list`: extracting nouns list
   > * `create_vocabulary`: creating custom vocabulary for topic model
   > * `get_topic_words`: generate topic model

## `preprocessing.py`

Main preprocessing functions for the Motor Vehicle Collisions dataset. In particular, include the following methods 
* `cleaning_pipeline`: decompose all columns in the original dataset into four groups and process each group accordingly. 
1.`drop_cols`: columns to be dropped  
2. `debatable_cols`: columns that are important and potentially should be kept (default to be dropped; keep by setting `drop_debatable=False`)
3. `basic_process_cols`: including most columns that only need to be "baically" processed, i.e. filling missing values and typos
4. `deep_process_cols`: columns that need special care (e.g. feature engineering -- groupping redundant features relying on specialty knowledge from HFD, performing frequency groupping on payroll numbers -- reducing highly distinct categories in nomial variable; for more details consult the docstrings in the `preprocessing.py` file).
     
* `encode_to_num_multiple`: label encoding function applicable to multiple columns

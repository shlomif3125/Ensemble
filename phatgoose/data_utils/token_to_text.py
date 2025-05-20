import re
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

DICT_MAP = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "|",
    "E",
    "T",
    "A",
    "O",
    "N",
    "I",
    "H",
    "S",
    "R",
    "D",
    "L",
    "U",
    "M",
    "W",
    "C",
    "F",
    "G",
    "Y",
    "P",
    "B",
    "V",
    "K",
    "'",
    "X",
    "J",
    "Q",
    "Z",
]

REGEX = re.compile("[^a-zA-Z' ]")


class IndicesToTextConverter:
    def __init__(self, dict_map: list = DICT_MAP, regex: re.Pattern = REGEX) -> None:
        self._dict_map = dict_map
        self._regex = regex

    def token_to_text(self, tokens: list) -> str:
        pred_ctc = "".join([self._dict_map[ii] for ii in tokens if ii != 1]).replace("|", " ").split()
        pred_ctc = " ".join(pred_ctc)
        pred_ctc = pred_ctc.replace("-", " ")
        pred_ctc = self._regex.sub("", pred_ctc).lower().strip()
        return pred_ctc

    def create_artifact_id_to_prediction_inline(self, df: pd.DataFrame, 
                                                tokens_column='logits_tokens', out_col_name='pred_text') -> pd.DataFrame:
        df[out_col_name] = df.logits_tokens.parallel_apply(
            self.token_to_text
        )
        return df



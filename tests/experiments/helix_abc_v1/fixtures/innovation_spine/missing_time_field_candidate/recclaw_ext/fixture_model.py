"""Generic candidate-local regression for a declared but absent time field."""

from recbole.model.general_recommender.bpr import BPR


class MissingDeclaredTimeFieldModel(BPR):
    """Reproduce a candidate-local field access without changing the adapter."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        time_field = getattr(dataset, "time_field", None)
        if time_field is not None:
            dataset.inter_feat[time_field]

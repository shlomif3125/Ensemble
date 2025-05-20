import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


__all__ = ['top_scoring_router_choiced_wer', 
           'thresholded_top_scoring_router_choiced_wer',
           'thresholded_weighted_router_choice_wer',
           'f1_metric', 'recall_metric', 'precision_metric',
           'roc_auc_metric']

def calc_wer(w_we_df):
        return w_we_df['we'].sum() / w_we_df['w'].sum()
    
    
def top_scoring_router_choiced_wer(results_df, *args, **kwargs):
    best_router_per_tar_id_df = results_df.loc[results_df.groupby('tar_id')['Score'].idxmax(), 
                                              ['instruction_type', 'w', 'we', 'baseline_we']]
    instruction_type_wers = best_router_per_tar_id_df.groupby('instruction_type').apply(calc_wer).to_dict()
    metric_name = 'TopScoringRouterChoice'
    return instruction_type_wers, metric_name
        

def thresholded_top_scoring_router_choiced_wer(results_df, router_score_threshold=0.5, *args, **kwargs):
    
    best_router_per_tar_id_df = results_df.loc[results_df.groupby('tar_id')['Score'].idxmax(), 
                                              ['instruction_type', 'Score', 'w', 'we', 'baseline_we']]
    best_router_per_tar_id_df['we'] = np.where(best_router_per_tar_id_df['Score'] > router_score_threshold,
                                               best_router_per_tar_id_df['we'],
                                               best_router_per_tar_id_df['baseline_we'])
    instruction_type_wers = best_router_per_tar_id_df.groupby('instruction_type').apply(calc_wer).to_dict()
    metric_name = 'ThresholdedTopScoringRouterChoice'
    return instruction_type_wers, metric_name


def thresholded_weighted_router_choice_wer(results_df, router_score_threshold=0.5, *args, **kwargs):
    
    def filter_by_thresh_and_weight_we_by_score(tar_id_rows):
        if (tar_id_rows['Score'] > router_score_threshold).any():
            tar_id_rows = tar_id_rows[tar_id_rows['Score'] > router_score_threshold]
            tar_id_rows['Score'] = tar_id_rows['Score'] / tar_id_rows['Score'].sum()
            tar_id_rows['we'] = (tar_id_rows['we'] * tar_id_rows['Score']).sum()
        else:
            tar_id_rows['we'] = tar_id_rows['baseline_we']
        return tar_id_rows.iloc[0]
    
    weighted_we_results_df = results_df.groupby('tar_id').apply(filter_by_thresh_and_weight_we_by_score)
    instruction_type_wers = weighted_we_results_df.groupby('instruction_type').apply(calc_wer).to_dict()
    metric_name = 'ThresholdedWeightedRouterChoice'
    return instruction_type_wers, metric_name


def f1_metric(results_df, score_threshold=0.5, *args, **kwargs):
    def get_f1_score(df):
        y_true = df['RouterLabel'].to_numpy().astype(np.int32)
        y_pred = (df['Score'].to_numpy() > score_threshold).astype(np.int32)
        return f1_score(y_true, y_pred)
    
    instruction_type_f1s = results_df.groupby('instruction_type').apply(get_f1_score).to_dict()
    metric_name = f'F1Score_{score_threshold}Thresh'
    return instruction_type_f1s, metric_name


def recall_metric(results_df, score_threshold=0.5, *args, **kwargs):
    def get_recall_score(df):
        y_true = df['RouterLabel'].to_numpy().astype(np.int32)
        y_pred = (df['Score'].to_numpy() > score_threshold).astype(np.int32)
        return recall_score(y_true, y_pred)
    
    instruction_type_recalls = results_df.groupby('instruction_type').apply(get_recall_score).to_dict()
    metric_name = f'RecallScore_{score_threshold}Thresh'
    return instruction_type_recalls, metric_name


def precision_metric(results_df, score_threshold=0.5, *args, **kwargs):
    def get_precision_score(df):
        y_true = df['RouterLabel'].to_numpy().astype(np.int32)
        y_pred = (df['Score'].to_numpy() > score_threshold).astype(np.int32)
        return precision_score(y_true, y_pred)
    
    instruction_type_precisions = results_df.groupby('instruction_type').apply(get_precision_score).to_dict()
    metric_name = f'PrecisionScore_{score_threshold}Thresh'
    return instruction_type_precisions, metric_name
        
        
def roc_auc_metric(results_df, score_threshold=0.5, *args, **kwargs):
    def get_roc_auc_score(df):
        y_true = df['RouterLabel'].to_numpy().astype(np.int32)
        y_score = df['Score'].to_numpy().astype(np.float32)
        return roc_auc_score(y_true, y_score)
    
    instruction_type_roc_aucs = results_df.groupby('instruction_type').apply(get_roc_auc_score).to_dict()
    metric_name = f'ROCAuCScore_{score_threshold}Thresh'
    return instruction_type_roc_aucs, metric_name
        


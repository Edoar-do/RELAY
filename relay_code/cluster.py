import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np 
import pandas as pd 
import wandb
import random
from tqdm import tqdm
import argparse
from sklearn.cluster import MiniBatchKMeans
import joblib
import sklearn
from consts import S, DIM_IT1, DIM_IT2
from loguru import logger
import torch

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.df = pd.read_csv(args.path_to_dataset_csv)['filename']
        self.filenames = self.df[int(args.start_perc * len(self.df)) : int(args.end_perc * len(self.df)) + 1]
        self.in_dir = args.in_dir
        self.train_iteration = args.train_iteration

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames.iloc[idx]
        if not os.path.isfile(filename):
            feature_path = os.path.join(self.in_dir, os.path.basename(filename).replace(".npy", "_grouped.npy"))
        else:
            feature_path = filename
        feature = np.load(feature_path, mmap_mode='r', allow_pickle=True) # 12, S, D or B, 12, S, D'
        # Flatten feature to [DIM_IT1] or [DIM_IT2] depending on iteration
        dim = DIM_IT1 if self.train_iteration == 1 else DIM_IT2
        feature = feature.reshape(-1, dim) # 12 * S, D or B * 12 * S, D'
        return torch.from_numpy(feature.copy())


def cluster(args):

    group = f"clustering_iteration_{args.train_iteration}"

    wandb.init(entity=None, project="RELAY", group=group)

    wandb.run.name = args.wandb_run_name if args.wandb_run_name is not None else wandb.run.name

    np.random.seed(42)
    random.seed(42)

    df = pd.read_csv(args.path_to_dataset_csv)['filename']
    df = df[int(args.start_perc * len(df)) : int(args.end_perc * len(df)) + 1]
    
    dataset = FeatureDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, prefetch_factor=4, shuffle=False, collate_fn=torch.stack)

    n_clusters = args.n_clusters_start
    
    normalizer = sklearn.preprocessing.Normalizer() # no need to fit this

    while n_clusters <= args.n_clusters_end:
        
        logger.info(f"Running kmeans with {n_clusters} clusters...")
        already_loaded = False
        
        if args.model_path is not None and already_loaded == False:
            logger.info(f"Loading model from {args.model_path}...")
            model = joblib.load(args.model_path)
            assert model.cluster_centers_.shape[0] == n_clusters, f"Loaded model has {model.cluster_centers_.shape[0]} clusters, but {n_clusters} were expected"
        else:
            logger.info("Creating a clustering model from scratch...")
        
            model = MiniBatchKMeans(
                n_clusters = n_clusters,
                random_state = 42,
                compute_labels = True,
                batch_size = args.batch_size * S * 12,
                n_init = 20,
                max_no_improvement = 100,
                reassignment_ratio = 0.0
            )
        
        ### FITTING LOOP ###
        
        for batched_features in tqdm(dataloader):
            
            batched_features = batched_features.reshape(-1, batched_features.size(-1))
            
            # normalize features in the batch
            features = normalizer.transform(batched_features.numpy())
            
            # train kmeans
            model.partial_fit(features) 
            
        ### END OF FITTING LOOP ###
            
        sse = model.inertia_
        wandb.log({"k" : n_clusters, "SSE" : sse})
        

        model_name = "k_means_" + str(n_clusters) + "_" + str(args.layer) + "_" + str(args.train_iteration)
            
        sse = "{:e}".format(int(sse))
        
        logger.info(f"SSE: {sse}")
        
        model_name +=  "_" + sse + ".pkl"
        
        joblib.dump(model, os.path.join(args.kmeans_models_path, model_name))
        logger.info(f"{model_name} model saved.\n")
        
        n_clusters = n_clusters + args.step
        
def evaluate_clustering(args):
    
    logger.info("Ready to evaluate clustering quality")
    
    np.random.seed(42)
    random.seed(42)
    
    group = f"clustering_iteration_{args.train_iteration}"

    wandb.init(entity=None, project="RELAY", group=group)

    wandb.run.name = args.wandb_run_name if args.wandb_run_name is not None else wandb.run.name
    
    dataset = FeatureDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, prefetch_factor=4, shuffle=False, collate_fn=torch.stack)
    
    kmeans_model = joblib.load(args.model_path)
    
    db_scores, ch_scores, sses = [], [], []
    
    normalizer = sklearn.preprocessing.Normalizer()
    
    for batch_features in tqdm(dataloader):
        
        batch_features = batch_features.reshape(-1, batch_features.size(-1)) # as long as the clustering dimension is the last one, this supports even batched inputs
        
        features = normalizer.transform(batch_features.numpy())
        
        labels = kmeans_model.predict(features)
        
        db_score = sklearn.metrics.davies_bouldin_score(features, labels)
        ch_score = sklearn.metrics.calinski_harabasz_score(features, labels)
        sses.append(kmeans_model.inertia_)
        
        db_scores.append(db_score)
        ch_scores.append(ch_score)
        
    mean_db = np.mean(db_scores)
    mean_ch = np.mean(ch_scores)
    sse = np.sum(sses)
    
    to_log = {
        "David-Bouldin score" : mean_db,
        "Calinski-Harabasz score" : mean_ch,
        "SSE" : sse,
        "E[SSE]" : sse / len(dataset)
    }

    wandb.log(to_log)
    
    logger.info(f"David-Bouldin score: {mean_db}")
    logger.info(f"Calinski-Harabasz score: {mean_ch}")
    logger.info(f"SSE: {sse}")
    logger.info(f"E[SSE]: {sse / len(dataset)}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Clustering argument parser")
    
    parser.add_argument(
        "--train_iteration",
        type=int,
        default=1,
        help="Pre-training iteration. Only for saving"
    )
    
    parser.add_argument(
        "--wandb_run_name", 
        help="wandb run name for monitoring. Default: randomly assigned",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--path_to_dataset_csv",
        help="path to the csv file referencing features to cluster",
        type=str,
    )
    
    parser.add_argument(
        "--start_perc",
        default=0.0,
        type=float,
        help="Starting point of the dataframe to read"
    )
    
    parser.add_argument(
        "--end_perc",
        default=1.0,
        type=float,
        help="Ending point of the dataframe to read"
    )
    
    parser.add_argument(
        "--in_dir",
        default=0.0,
        type=str,
        help="Path to the input directory containing the features to cluster"
    )
    
    parser.add_argument(
        "--n_clusters_start",
        type=int,
        help="Number of cluster to find initially"
    )
    
    parser.add_argument(
        "--n_clusters_end",
        type=int,
        help="Number of cluster to find eventually"
    )
    
    parser.add_argument(
        "--step",
        type=int,
        help="Number of clusters between start and end",
        default=100
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of grouped features (12, S, DIM) to load. Default 100"
    )
    
    parser.add_argument(
        "--layer",
        type=int,
        help="Feature extraction layer. Deafult 0 (for iteration 1)",
        default=0
    )
    
    parser.add_argument(
        "--kmeans_models_path",
        type=str,
        help="Path to the location where to save the kmeans model",
        default="./kmeans_models/"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to a kmeans model to load. Default None",
        default=None
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="If set, the script will evaluate the clustering quality of the model at model_path instead of training a new one"
    )
    
    args = parser.parse_args()
    
    # --- checks ---   
    
    assert args.train_iteration >= 1, f"train_iteration must be >= 1, got {args.train_iteration}"
    
    assert args.start_perc < args.end_perc and args.start_perc >= 0 and args.end_perc <= 1, f"0 <= start_perc < end_perc <= 1, got {args.start_perc} and {args.end_perc}"
    
    assert os.path.exists(args.path_to_dataset_csv), f"{args.path_to_dataset_csv} does not exists"
    
    assert os.path.exists(args.in_dir), f"{args.in_dir} does not exists"
    
    if args.train_iteration == 1:
        assert args.layer == 0, f"Cannot set layer to a value different from 0 if train_iteration is 1. Got {args.layer}"
        
    print("Inserted arguments: ")
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
        
    if args.evaluate:
        assert args.model_path is not None, "To evaluate clustering quality, model_path must be specified"
        assert os.path.exists(args.model_path), f"{args.model_path} does not exists"
        evaluate_clustering(args)
    else:
        assert args.n_clusters_start <= args.n_clusters_end, f"n_clusters_start {args.n_clusters_start} must be <= of n_clusters_end {args.n_clusters_end}"
        assert os.path.exists(args.kmeans_models_path), f"{args.kmeans_models_path} does not exists"
        assert os.path.isdir(args.kmeans_models_path), f"{args.kmeans_models_path} is not a directory"
        cluster(args)
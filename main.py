import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import random
import numpy as np
import torch
from Rewriter import CommRewriting
from Locator import CommMatching
from utils import split_communities, eval_scores, prepare_data
import os
import networkx as nx
from networkx.algorithms.community.quality import modularity
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def seed_all(seed):
    """
    Set random seed for reproducibility across numpy, random, and torch.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write2file(comms, filename):
    """
    Write communities to a file for storage.
    """
    with open(filename, 'w') as fh:
        content = '\n'.join([', '.join([str(i) for i in com])
                            for com in comms])
        fh.write(content)


def read4file(filename):
    """
    Read communities from a file.
    """
    with open(filename, "r") as file:
        pred = [[int(node) for node in x.split(', ')]
                for x in file.read().strip().split('\n')]
    return pred


def filter_noisy_communities(communities, nx_graph, modularity_threshold=-0.01):
    """
    Filters out noisy communities based on a modularity score threshold.
    Communities with modularity scores below the given threshold are excluded.

    Args:
        communities (list of lists): A list of communities, where each community is a list of nodes.
        nx_graph (networkx.Graph): A NetworkX graph representing the dataset.
        modularity_threshold (float): The minimum modularity score a community must have to be retained.
                                      Default is -0.01.

    Returns:
        list of lists: A list of communities that meet the modularity threshold criteria.
    """
    # Initialize a list to store valid communities that meet the modularity threshold
    valid_communities = []

    # Loop through each community in the provided list of communities
    for i, community in enumerate(communities):
        # Skip communities with size <= 1 (single-node or empty communities)
        if len(community) <= 1:
            print(
                f"Skipping community {i} with insufficient size: {community}")
            continue  # Move to the next community in the list

        try:
            # Create a partition to calculate modularity:
            # 1. First part contains the nodes in the current community
            # 2. Second part contains all other nodes in the graph (not in the current community)
            partition = [set(community)] + \
                [set(nx_graph.nodes) - set(community)]

            # Calculate the modularity of the partition for the current community
            mod = modularity(nx_graph, partition)

            # Debugging: Uncomment the following line to see the modularity of each community
            # print(f"Community {i}: Modularity = {mod}")

            # If the modularity is greater than or equal to the threshold, keep the community
            if mod >= modularity_threshold:
                valid_communities.append(community)

        # Handle exceptions during modularity calculation (e.g., due to invalid graph structure or input)
        except Exception as e:
            print(f"Error calculating modularity for community {i}: {e}")

    # Print the number of noisy communities filtered out and the total number of communities processed
    print(
        f"Filtered {len(communities) - len(valid_communities)} noisy communities out of {len(communities)}.")

    # Return the list of valid communities that meet the modularity threshold
    return valid_communities


def plot_modularity_distribution(modularity_scores):
    """
    Plot a histogram to visualize the distribution of modularity scores.
    """
    plt.hist(modularity_scores, bins=20, edgecolor='black')
    plt.title("Modularity Score Distribution")
    plt.xlabel("Modularity Score")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modularity_threshold", type=float,
                        help="Minimum modularity score to retain a community", default=-0.01)

    # General Config
    parser.add_argument("--seed", type=int, help="seed", default=0)
    parser.add_argument("--device", dest="device", type=str,
                        help="training device", default="cuda:0")
    parser.add_argument("--dataset", type=str,
                        help="dataset", default="amazon")
    #   --in CLARE paper, we predict 1000 communities from 100 communities as a default setting
    parser.add_argument("--num_pred", type=int, help="pred size", default=1000)
    parser.add_argument("--num_train", type=int, help="pred size", default=90)
    parser.add_argument("--num_val", type=int, help="pred size", default=10)

    # Community Locator related
    #   --GNNEncoder Setting
    parser.add_argument("--gnn_type", type=str,
                        help="type of convolution", default="GCN")
    parser.add_argument("--n_layers", type=int,
                        help="number of gnn layers", default=2)
    parser.add_argument("--hidden_dim", type=int,
                        help="training hidden size", default=64)
    parser.add_argument("--output_dim", type=int,
                        help="training hidden size", default=64)
    #   --Order Embedding Setting
    parser.add_argument("--margin", type=float,
                        help="margin loss", default=0.6)
    #   --Generation
    parser.add_argument("--comm_max_size", type=int,
                        help="Community max size", default=12)
    #   --Training
    parser.add_argument("--locator_lr", type=float,
                        help="learning rate", default=1e-3)
    parser.add_argument("--locator_epoch", type=int, default=30)
    parser.add_argument("--locator_batch_size", type=int, default=256)

    # Community Rewriter related
    parser.add_argument("--agent_lr", type=float,
                        help="CommR learning rate", default=1e-3)
    #    -- for DBLP, the setting of n_eisode and n_epoch is a little picky
    parser.add_argument("--n_episode", type=int,
                        help="number of episode", default=10)
    parser.add_argument("--n_epoch", type=int,
                        help="number of epoch", default=1000)
    parser.add_argument("--gamma", type=float,
                        help="CommR gamma", default=0.99)
    parser.add_argument("--max_step", type=int, help="", default=10)
    parser.add_argument("--max_rewrite_step", type=int, help="", default=4)
    parser.add_argument("--commr_path", type=str,
                        help="CommR path", default="")

    # Save log
    parser.add_argument("--writer_dir", type=str,
                        help="Summary writer directory", default="")

    args = parser.parse_args()
    seed_all(args.seed)

    if not os.path.exists(f"ckpts/{args.dataset}"):
        os.mkdir(f"ckpts/{args.dataset}")
    args.writer_dir = f"ckpts/{args.dataset}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.mkdir(args.writer_dir)
    args.comm_max_size = 20 if args.dataset.startswith("lj") else 12

    print('= ' * 20)
    print('##  Starting Time:', datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"), flush=True)
    print(args)


##########################################################
################### Step 1 Load Data #####################
##########################################################
# Prepare the data for analysis and model training
# - num_node: Number of nodes in the graph
# - num_edge: Number of edges in the graph
# - num_community: Number of communities in the dataset
# - graph_data: Graph data structure prepared for further analysis
# - nx_graph: NetworkX graph representation of the dataset
# - communities: List of communities from the dataset
num_node, num_edge, num_community, graph_data, nx_graph, communities = prepare_data(
    args.dataset)
print(f"Finish loading data: {graph_data}\n")

##########################################################
############### Step 2 Modularity Analysis ###############
##########################################################
# Initialize an empty list to store modularity scores of communities
modularity_scores = []

# Loop through each community to calculate its modularity score
for community in communities:
    # Skip trivial or single-node communities as modularity is not meaningful for them
    if len(community) > 1:
        partition = [set(community)] + [set(nx_graph.nodes) -
                                        set(community)]  # Partition graph nodes
        try:
            mod = modularity(nx_graph, partition)  # Calculate modularity
            modularity_scores.append(mod)  # Append the score to the list
        except Exception as e:
            print(f"Error calculating modularity for a community: {e}")

# Plot the distribution of modularity scores for visualization
print("Plotting modularity distribution...")
plot_modularity_distribution(modularity_scores)

##########################################################
############# Step 3 Filter Noisy Communities ############
##########################################################
# Print the number of communities before filtering
print(f"Before filtering: Total communities: {len(communities)}")

# Filter out communities with modularity scores below the threshold
communities = filter_noisy_communities(
    communities, nx_graph, modularity_threshold=args.modularity_threshold)

# Handle the case where all communities are filtered out
if len(communities) == 0:
    print("All communities were filtered. Reverting to the original dataset.")
    # Reload the original dataset
    num_node, num_edge, num_community, graph_data, nx_graph, communities = prepare_data(
        args.dataset)

##########################################################
################### Step 4 Split Dataset #################
##########################################################
# Split the communities into training, validation, and test sets
# - train_comms: Communities for training
# - val_comms: Communities for validation
# - test_comms: Communities for testing
train_comms, val_comms, test_comms = split_communities(
    communities, args.num_train, args.num_val)

# Print details about the dataset after filtering and splitting
print(
    f"After filtering: Total communities: {len(train_comms) + len(val_comms) + len(test_comms)}")
print(
    f"Split dataset: #Train {len(train_comms)}, #Val {len(val_comms)}, #Test {len(test_comms)}\n")

##########################################################
################### Step 5 Train Locator #################
##########################################################
# Initialize the community matching (locator) object
CommM_obj = CommMatching(
    args, graph_data, train_comms, val_comms, device=torch.device(args.device))

# Train the locator model on the training dataset
CommM_obj.train()

# Predict communities using the trained locator model
pred_comms = CommM_obj.predict_community(nx_graph, args.comm_max_size)

# Evaluate the predicted communities using F1 Score, Jaccard Index, and NMI
f1, jaccard, onmi = eval_scores(pred_comms, test_comms, tmp_print=True)

# Save the predicted communities and their evaluation metrics to a file
metrics_string = '_'.join([f'{x:0.4f}' for x in [f1, jaccard, onmi]])
write2file(pred_comms, args.writer_dir +
           "/CommM_" + metrics_string + '.txt')

##########################################################
################### Step 6 Train Rewriter ################
##########################################################
# Set the cost function for training the rewriter (e.g., "f1" or "jaccard")
cost_choice = "f1"

# Generate node embeddings from the trained locator model
feat_mat = CommM_obj.generate_all_node_emb(
).detach().cpu().numpy()  # Get node embeddings as a NumPy array

# Initialize the community rewriter object
CommR_obj = CommRewriting(args, nx_graph, feat_mat,
                          train_comms, val_comms, pred_comms, cost_choice)

# Train the rewriter model on the training dataset
CommR_obj.train()

# Get the rewritten (refined) communities from the rewriter model
rewrite_comms = CommR_obj.get_rewrite()

# Evaluate the rewritten communities using F1 Score, Jaccard Index, and NMI
f1, jaccard, onmi = eval_scores(rewrite_comms, test_comms, tmp_print=True)

# Save the rewritten communities and their evaluation metrics to a file
metrics_string = '_'.join([f'{x:0.4f}' for x in [f1, jaccard, onmi]])
write2file(rewrite_comms, args.writer_dir +
           f"/CommR_{cost_choice}_" + metrics_string + '.txt')

# Print the finishing time of the process
print('## Finishing Time:', datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"), flush=True)
print('= ' * 20)

import numpy as np
import torch


def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type):
    """
    Computes the similarity matrix using pooled video frames

    Output
        sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type == "avg":
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())

    else:
        # num_texts x embed_dim x num_vids
        vid_embeds_pooled = vid_embeds_pooled.permute(1, 2, 0)
        # num_texts x 1 x embed_dim
        text_embeds = text_embeds.unsqueeze(1)

        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims


def sim_matrix_inference(
    text_embeds_per_video_id, vid_embeds_pooled_per_video_id, pooling_type
):
    """
    Computes the similarity matrix using pooled video frames using all texts per video

    Output
        sims: num_vids x max_text_per_vid x num_vids
    """
    text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(
        dim=-1, keepdim=True
    )
    vid_embeds_pooled_per_video_id = (
        vid_embeds_pooled_per_video_id
        / vid_embeds_pooled_per_video_id.norm(dim=-1, keepdim=True)
    )

    if pooling_type == "avg":
        # text_embeds_per_video_id -> num_vids x max_text_per_vid x embed_dim
        # vid_embeds_pooled_per_video_id -> num_vids x embed_dim

        sims = text_embeds_per_video_id @ vid_embeds_pooled_per_video_id.t()

    else:
        # text_embeds_per_video_id -> num_vids x max_text_per_vid x embed_dim
        # vid_embeds_pooled_per_video_id -> num_vids x num_vids x max_text_per_vid x embed_dim
        num_vids, max_text_per_vid, embed_dim = text_embeds_per_video_id.shape

        # num_vids x max_text_per_vid x embed_dim x num_vids
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.permute(
            1, 2, 3, 0
        )
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.view(
            num_vids * max_text_per_vid, embed_dim, num_vids
        )
        # num_vids x max_text_per_vid x 1 x embed_dim
        text_embeds_per_video_id = text_embeds_per_video_id.unsqueeze(2)
        text_embeds_per_video_id = text_embeds_per_video_id.view(
            num_vids * max_text_per_vid, 1, embed_dim
        )

        sims = torch.bmm(text_embeds_per_video_id, vid_embeds_pooled_per_video_id)
        sims = sims.view(num_vids, max_text_per_vid, 1, num_vids).squeeze(2)

    return sims

def compute_labeled_metrics(sims, labels):
    """
    Compute new evaluation metrics based on the given similarity matrix and labels
    """
    num_vids, _ = sims.shape

    new_metrics = {"new_R1": 0, "new_R3": 0, "new_R5": 0, "new_P1": 0, "new_P3": 0, "new_P5": 0}
    total_queries = 0

    # Pre-compute all video label sets for efficiency
    v_label_sets = [set(label.split("-")) for label in labels]

    skipping_labels = set()
    not_good_labels = set()

    y_true = []
    y_pred = []

    for i in range(num_vids):
        # Convert the query label to a set
        q_labels = v_label_sets[i]

        # Calculate total relevant videos for this query
        total_relevant = sum(1 for v_labels in v_label_sets if q_labels.issubset(v_labels))

        if total_relevant <= 6:
            skipping_labels.add("-".join(sorted(q_labels)))
            continue  # Skip if no relevant videos

        # Get the similarity scores for the current text query
        sim_scores = sims[i].cpu().numpy().copy()
        # Get the top 5 retrieved video indices
        top_indices = np.argsort(-sim_scores)[:5]

        # Check relevance for top retrieved videos
        is_relevant_list = []
        for idx in top_indices:
            retrieved_v_labels = v_label_sets[idx]
            is_relevant = q_labels.issubset(retrieved_v_labels)
            is_relevant_list.append(is_relevant)

        if sum(is_relevant_list) <= 1:
            not_good_labels.add("-".join(sorted(q_labels)))
            continue  # Skip if no relevant videos in top 5

        # Calculate Recall@k (existing logic)
        new_metrics["new_R1"] += any(is_relevant_list[:1])
        new_metrics["new_R3"] += any(is_relevant_list[:3])
        new_metrics["new_R5"] += any(is_relevant_list[:5])

        # Calculate Precision@k
        new_metrics["new_P1"] += sum(is_relevant_list[:1]) / 1
        new_metrics["new_P3"] += sum(is_relevant_list[:3]) / 3
        new_metrics["new_P5"] += sum(is_relevant_list[:5]) / 5

        total_queries += 1

        y_true.append("-".join(sorted(q_labels)))
        y_pred.append("-".join(sorted(v_label_sets[top_indices[0]])))

    with open("labeled_classification_results.txt", "a") as f:
        for true_label, pred_label in zip(y_true, y_pred):
            f.write(f"{true_label}|{pred_label}\n")
    # print("Skipping labels due to insufficient relevant videos: ", skipping_labels)
    # print("Length of skipping labels: ", len(skipping_labels))
    # print("Not good labels with no relevant videos in top 5: ", not_good_labels)
    # print("Length of not good labels: ", len(not_good_labels))

    # Normalize the metrics
    if total_queries > 0:
        new_metrics["new_R1"] = new_metrics["new_R1"] * 100 / total_queries
        new_metrics["new_R3"] = new_metrics["new_R3"] * 100 / total_queries
        new_metrics["new_R5"] = new_metrics["new_R5"] * 100 / total_queries

        new_metrics["new_P1"] = new_metrics["new_P1"] * 100 / total_queries
        new_metrics["new_P3"] = new_metrics["new_P3"] * 100 / total_queries
        new_metrics["new_P5"] = new_metrics["new_P5"] * 100 / total_queries
    else:
        new_metrics["new_R1"] = new_metrics["new_R3"] = new_metrics["new_R5"] = 0.0
        new_metrics["new_P1"] = new_metrics["new_P3"] = new_metrics["new_P5"] = 0.0

    return new_metrics


def generate_embeds_per_video_id(text_embeds, vid_embeds_pooled, video_ids, pooling_type):
    # Construct dictionary of text embeds per unique video id
    video_id_to_text_embeds = {}

    for idx, v_id in enumerate(video_ids):
        video_id_to_text_embeds.setdefault(v_id, [])
        video_id_to_text_embeds[v_id].append(text_embeds[idx])

    video_id_to_text_embeds = {video_id: torch.stack(text_embeds) for video_id, text_embeds in video_id_to_text_embeds.items()}

    # num_vids x max_text_per_vid x embed_dim
    video_id_to_text_embeds = pad_and_stack_dict_to_tensor(video_id_to_text_embeds, video_id_to_text_embeds.keys(), text_embeds.shape[-1])

    if pooling_type == "avg":
        # num_vids x embed_dim
        vid_embeds_pooled_per_video_id = vid_embeds_pooled

    else:
        # Construct dictionary of video embeds for each text per video_id
        vid_embeds_pooled_per_video_id = []

        for i in range(vid_embeds_pooled.shape[0]):
            vid_embeds_pooled_per_video_id.append({})
            for idx, v_id in enumerate(video_ids):
                if v_id in vid_embeds_pooled_per_video_id[i]:
                    vid_embeds_pooled_per_video_id[i][v_id].append(vid_embeds_pooled[i, idx, :])
                else:
                    vid_embeds_pooled_per_video_id[i][v_id] = [vid_embeds_pooled[i, idx, :]]

        for i in range(len(vid_embeds_pooled_per_video_id)):
            for v_id in vid_embeds_pooled_per_video_id[i]:
                vid_embeds_pooled_per_video_id[i][v_id] = torch.stack(vid_embeds_pooled_per_video_id[i][v_id])

            # num_vids x max_text_per_vid x embed_dim
            vid_embeds_pooled_per_video_id[i] = pad_and_stack_dict_to_tensor(
                vid_embeds_pooled_per_video_id[i],
                vid_embeds_pooled_per_video_id[i].keys(),
                vid_embeds_pooled.shape[-1],
            )

        # num_vids x num_vids x max_text_per_vid x embed_dim
        vid_embeds_pooled_per_video_id = torch.stack(vid_embeds_pooled_per_video_id)

    return video_id_to_text_embeds, vid_embeds_pooled_per_video_id



def t2v_metrics(sims):
    # Permute sims so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sims = sims.permute(1, 0, 2)

    sims_sort = torch.argsort(stacked_sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.flatten(torch.diagonal(sims_sort_2, dim1=1, dim2=2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    valid_check = torch.flatten(torch.diagonal(sims, dim1=0, dim2=2))
    mask = ~torch.logical_or(torch.isinf(valid_check), torch.isnan(valid_check))
    valid_ranks = ranks[mask]

    return compute_metrics(valid_ranks.numpy())


def v2t_metrics(sims):
    # Code to avoid nans
    sims[sims != sims] = float("-inf")
    # Forms a similarity matrix
    sims, _ = torch.max(sims, dim=1)
    sims = sims.t()

    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.diag(sims_sort_2).numpy()  # diagonal

    return compute_metrics(ranks)


def compute_metrics(lst):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(lst == 0)) / len(lst)
    metrics["R3"] = 100 * float(np.sum(lst < 3)) / len(lst)
    metrics["R5"] = 100 * float(np.sum(lst < 5)) / len(lst)
    return metrics


def pad_and_stack_dict_to_tensor(input, order, d=512):
    max_length = max([input[k].shape[0] for k in input])

    padded_input = {
        k: torch.cat(
            [
                input[k],
                torch.full(
                    (max_length - input[k].shape[0], d),
                    float("-inf"),
                    device=input[k].device,
                ),
            ]
        )
        for k in input
    }

    padded_stacked_input = torch.stack([padded_input[k] for k in order], dim=0)
    return padded_stacked_input

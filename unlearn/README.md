# ✨Baselines 

## GA

This section describes how to run the **GA** (Gradient Ascent) baseline. GA performs selective forgetting by optimizing the model against a target forget set.

### ✅ Usage

To execute the GA baseline, run the following command:

```bash
python GA.py \
    --vanilla_dir <path_to_vanilla_model> \
    --save_dir <path_to_save_forget_model> \
    --data_split_dir <path_to_data_split> \
    --forget_split_ratio <forget_ratio> \
    --batch_size <batch_size> \
    --alpha <alpha_value> \
    --lr <learning_rate> \
    --num_epochs <num_epochs>
```

📁 **Output**

After training, the model will be saved to the directory specified by `--save_dir`. You can load the saved model using a standard inference script and evaluate its performance on both the retain and forget sets to verify the effectiveness of unlearning.

## GD

This section provides instructions on running the **GD** baseline. Gradient Difference is an unlearning method that minimizes loss on retained data while maximizing loss on data to forget, effectively balancing forgetting and retention.

To execute the GD baseline, run the following command:

```bash
python GD.py \
    --vanilla_dir <path_to_vanilla_model> \
    --save_dir <path_to_save_forget_model> \
    --data_split_dir <path_to_data_split> \
    --gamma <gamma_value> \
    --forget_split_ratio <forget_ratio> \
    --batch_size <batch_size> \
    --alpha <alpha_value> \
    --lr <learning_rate> \
    --num_epochs <num_epochs>
```

## KL

This section provides instructions on running the **KL** baseline. KL Minimization is an unlearning method that enforces forgetting by maximizing task loss on the forget set while preserving knowledge by minimizing KL divergence between the current and original model on the retain set.

### ✅ Usage

To execute the KL baseline, run the following command:

```bash
python KL.py \
    --vanilla_dir <path_to_vanilla_model> \
    --oracle_model_id <oracle_model_id> \
    --save_dir <path_to_save_forget_model> \
    --data_split_dir <path_to_data_split> \
    --forget_split_ratio <forget_ratio> \
    --batch_size <batch_size> \
    --alpha <alpha_value> \
    --gamma <gamma_value> \
    --lr <learning_rate> \
    --num_epochs <num_epochs>
```

## NPO

This section provides instructions on running the **NPO** baseline. NPO (Negative Preference Optimization) is an unlearning method that penalizes the model for assigning high probability to original labels in the forget set, using a reference distribution (e.g., uniform) to encourage uncertainty, without relying on a retain set or original model.

### ✅ Usage

To execute the NPO baseline, run the following command:

```bash
python NPO.py \
    --vanilla_dir <path_to_vanilla_model> \
    --oracle_model_id <oracle_model_id> \
    --save_dir <path_to_save_forget_model> \
    --data_split_dir <path_to_data_split> \
    --forget_split_ratio <forget_ratio> \
    --batch_size <batch_size> \
    --alpha <alpha_value> \
    --beta <beta_value> \
    --lr <learning_rate> \
    --num_epochs <num_epochs>
```

## PO

This section provides instructions on running the **PO** baseline. PO (Preference Optimization) is an unlearning method that replaces sensitive responses in the forget set with refusals (e.g., “I don't know”), training the model to prefer refusals over revealing content, while preserving general performance on the retain set.

### ✅ Usage

To execute the PO baseline, run the following command:

```bash
python PO.py \
    --vanilla_dir <path_to_vanilla_model> \
    --save_dir <path_to_save_forget_model> \
    --data_split_dir <path_to_data_split> \
    --gamma <gamma_value> \
    --forget_split_ratio <forget_ratio> \
    --batch_size <batch_size> \
    --alpha <alpha_value> \
    --lr <learning_rate> \
    --num_epochs <num_epochs>
```
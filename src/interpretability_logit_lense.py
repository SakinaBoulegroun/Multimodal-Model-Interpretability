import torch

def get_logit_lens_outputs(model, classifier, inputs, layers, device):
    """
    Collects intermediate hidden states from specified layers, applies the classifier 
    head on the [CLS] token, and returns the logits per layer.
    
    Args:
        model (nn.Module): The full transformer model (e.g., a multimodal or encoder-only model).
        classifier (nn.Module): The classifier head to apply on the CLS representation.
        inputs (dict): Dictionary containing model inputs (e.g., input_ids, attention_mask, etc.).
        layers (list): List of layer modules to register hooks on.
        device (torch.device): Device to run the model on.

    Returns:
        list of tuples: [(layer_name, logits), ...]
    """
    hidden_states_dict = {}

    # Hook registration function
    def get_hook(layer_idx):
        def save_hidden_state(module, input, output):
            key = f"layer {layer_idx}"
            hidden_states_dict[key] = output[0].detach() if isinstance(output, tuple) else output.detach()
        return save_hidden_state

    # Register hooks
    hooks = [layer.register_forward_hook(get_hook(i)) for i, layer in enumerate(layers)]

    # Run model to populate hidden states
    with torch.no_grad():
        model.eval()
        _ = model(**{k: v.unsqueeze(0).to(device) for k, v in inputs.items() if k != "labels"})


    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute logits from each saved hidden state (using [CLS] token)
    logit_lens_outputs = []
    for name, h in hidden_states_dict.items():
        cls_token = h[:, 0, :]  # CLS token that gives a summary of the sentence that comes next
        logits = classifier(cls_token)
        logit_lens_outputs.append((name, logits.cpu()))

    return logit_lens_outputs

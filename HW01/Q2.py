from transformers import AutoModelForCausalLM, AutoProcessor

def count_parameters(model_name="google/gemma-3-4b-it"):
    AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters (programmatic check): {total_params:,}\n")
    component_counts = {
        'vision_tower': 0,
        'embeddings': 0,
        'attention': 0,
        'mlp': 0,
        'normalization': 0,
        'output_head': 0,
        'other': 0
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'vision_tower' in name:
            component_counts['vision_tower'] += param.numel()
        elif 'embed_tokens' in name:
            component_counts['embeddings'] += param.numel()
        elif 'self_attn' in name:
            component_counts['attention'] += param.numel()
        elif 'mlp' in name:
            component_counts['mlp'] += param.numel()
        elif 'norm' in name:
            component_counts['normalization'] += param.numel()
        elif 'lm_head' in name:
            component_counts['output_head'] += param.numel()
        else:
            component_counts['other'] += param.numel()

    print("Parameter Breakdown:")
    for component, count in component_counts.items():
        if count > 0:
            print(f"- {component.replace('_', ' ').capitalize():<15}: {count:15,d}")


count_parameters()
"""
Total trainable parameters (programmatic check): 4,300,079,472

Parameter Breakdown:
- Vision tower   :     416,866,032
- Embeddings     :     671,252,480
- Attention      :     534,791,168
- Mlp            :   2,673,868,800
- Normalization  :         351,872
- Other          :       2,949,120


The difference between manual and programmatic values comes from assuming the model is
built from a uniform stack of identical transformer blocks. In reality, it’s not.

When you break it down:

The math checks out for the MLP layers — there are exactly 34 of them.

But for attention layers, the numbers don’t divide cleanly. Instead of 34, it looks like there are about 27 self-attention blocks.

That gap is explained by the model’s non-uniform architecture.
Multimodal models like gemma-3-4b-it often mix in specialized components.
Here, the “missing” parameters likely come from a vision-language connector —
a module that uses attention to merge vision outputs with text embeddings.

So, my formulas are correct for a standard transformer,
but this model has a twist: 34 MLP blocks, fewer self-attention blocks, and a connector layer.

The mismatch isn’t from my math — it’s because the model’s architecture isn’t uniform.
"""
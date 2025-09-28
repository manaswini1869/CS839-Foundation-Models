from transformers import GPT2Model
# model = GPT2Model.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2-medium')
# model = GPT2Model.from_pretrained('gpt2-large')
model = GPT2Model.from_pretrained('gpt2-xl')

def count_params(model, is_human: bool = False):
    params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{params / 1e6:.2f}M" if is_human else params

print(model)
# printing the model params
print("Total # of params:", count_params(model, is_human=True))

# # wpe and wte
# # The embeddings from these two layers will get added together to create “position-aware” embeddings of our input tokens.
# # Let’s verify our math with some code:

# V: int = model.config.vocab_size
# E: int = model.config.n_embd
# P: int = model.config.n_positions
# expected_wte = V * E
# expected_wpe: int = P * E
# print("position-aware embeddings")
# print(f"wte | Expected: {expected_wte}")
# print(f"wte | True:     {count_params(model._modules['wte'])}")
# print(f"wpe | Expected: {expected_wpe}")
# print(f"wpe | True:     {count_params(model._modules['wpe'])}")

# # ln_1 params formula : 2 * E
# expected_ln_1 = 2 * E
# print(f"ln_1 | Expected: {expected_ln_1}")
# print(f"ln_1 | True:     {count_params(model._modules['h'][0].ln_1)}")

# # attn Params = c_attn + c_proj + attn_dropout + resid_dropout
# expected_c_attn = E * (3 * E) + (3 * E)
# expected_c_proj = E * E + E
# expected_attn_dropout = 0
# expected_resid_dropout = 0
# expected_attn = expected_c_attn + expected_c_proj + expected_attn_dropout + expected_resid_dropout
# print(f"c_attn | Expected: {expected_c_attn}")
# print(f"c_attn | True:     {count_params(model._modules['h'][0].attn.c_attn)}")
# print(f"c_proj | Expected: {expected_c_proj}")
# print(f"c_proj | True:     {count_params(model._modules['h'][0].attn.c_proj)}")
# print(f"attn_dropout | Expected: {expected_attn_dropout}")
# print(f"attn_dropout | True:     {count_params(model._modules['h'][0].attn.attn_dropout)}")
# print(f"resid_dropout | Expected: {expected_resid_dropout}")
# print(f"resid_dropout | True:     {count_params(model._modules['h'][0].attn.resid_dropout)}")
# print(f"attn | Expected: {expected_attn}")
# print(f"attn | True:     {count_params(model._modules['h'][0].attn)}")

# # ln_2 Params = 2∗E
# expected_ln_2 = 2 * E
# print(f"ln_2 | Expected: {expected_ln_2}")
# print(f"ln_2 | True:     {count_params(model._modules['h'][0].ln_2)}")

# # mlp
# # Params = c_fc + c_proj + act + dropout

# H: int = 4 * E
# expected_c_fc = E * H + H
# expected_c_proj = H * E + E
# expected_act = 0
# expected_dropout = 0
# expected_mlp = expected_c_fc + expected_c_proj + expected_act + expected_dropout
# print(f"c_fc | Expected: {expected_c_fc}")
# print(f"c_fc | True:     {count_params(model._modules['h'][0].mlp.c_fc)}")
# print(f"c_proj | Expected: {expected_c_proj}")
# print(f"c_proj | True:     {count_params(model._modules['h'][0].mlp.c_proj)}")
# print(f"act | Expected: {expected_act}")
# print(f"act | True:     {count_params(model._modules['h'][0].mlp.act)}")
# print(f"dropout | Expected: {expected_dropout}")
# print(f"dropout | True:     {count_params(model._modules['h'][0].mlp.dropout)}")
# print(f"mlp | Expected: {expected_mlp}")
# print(f"mlp | True:     {count_params(model._modules['h'][0].mlp)}")

# #ln_f
# # Params = 2∗E

# expected_ln_f = 2 * E
# print(f"ln_f | Expected: {expected_ln_f}")
# print(f"ln_f | True:     {count_params(model._modules['ln_f'])}")

# ## Final formula for the parameter count in GPT-2

# # C = embed_layers+transformer_layers+other
# #   =(wte+wpe)+L∗(ln_1+attn+ln_2+mlp)+ln_f
# #   =(V E + P E) + L(2E + (4E^2 + 4E) + 2E + (2EH + E + H ))+(2E)
# #   =E(V+P)+L(12E^2 + 13E)+2E

# L: int = model.config.n_layer
# expected_gpt2: int = E * (V + P) + L * (12 * E * E + 13 * E) + (2 * E)
# print(f"gpt2 | Expected: {expected_gpt2}")
# print(f"gpt2 | True:     {count_params(model)}")
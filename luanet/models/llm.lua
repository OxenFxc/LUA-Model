local Module = require("luanet.module")
local Tensor = require("luanet.tensor")
local layers = require("luanet.layers.init")

local TransformerBlock = setmetatable({}, {__index = Module})
TransformerBlock.__index = TransformerBlock

function TransformerBlock:new(embed_dim, num_heads, dropout_p)
    local self = setmetatable({}, TransformerBlock)
    self:init()

    self.ln1 = layers.LayerNorm:new(embed_dim)
    self.attn = layers.MultiHeadAttention:new(embed_dim, num_heads)
    self.dropout1 = layers.Dropout:new(dropout_p)

    self.ln2 = layers.LayerNorm:new(embed_dim)

    self.ffn = layers.Sequential:new(
        layers.Linear:new(embed_dim, 4 * embed_dim),
        layers.GELU:new(),
        layers.Linear:new(4 * embed_dim, embed_dim),
        layers.Dropout:new(dropout_p)
    )

    self.dropout2 = layers.Dropout:new(dropout_p)

    self:add_module("ln1", self.ln1)
    self:add_module("attn", self.attn)
    self:add_module("dropout1", self.dropout1)
    self:add_module("ln2", self.ln2)
    self:add_module("ffn", self.ffn)
    self:add_module("dropout2", self.dropout2)

    return self
end

function TransformerBlock:forward(x, mask)
    self.input = x

    local h = self.ln1:forward(x)
    local attn_out = self.attn:forward(h, h, h, mask)
    attn_out = self.dropout1:forward(attn_out)

    self.res1 = x + attn_out

    local h2 = self.ln2:forward(self.res1)

    -- Flatten for FFN
    local B = h2.shape[1]
    local S = h2.shape[2]
    local D = h2.shape[3]
    local h2_flat = h2:view({B * S, D})

    local ffn_out_flat = self.ffn:forward(h2_flat)
    local ffn_out = ffn_out_flat:view({B, S, D})

    -- No need to call dropout2 explicitly here if it's inside FFN?
    -- Wait, FFN has dropout at end? Yes.
    -- But usually dropout is applied to the residual connection input.
    -- My ffn definition includes dropout at end.
    -- But standard Transformer: x = x + Dropout(FFN(LN(x)))
    -- My FFN ends with Dropout. So ffn_out is already dropped.
    -- However, I added `self.dropout2` but didn't use it in FFN definition above optimally.
    -- Let's change FFN definition to NOT include the final dropout, and apply it here.
    -- Actually, my FFN above has `layers.Dropout:new(dropout_p)`.
    -- So `ffn_out` is already dropped.
    -- `self.dropout2` is redundant if I put it in `ffn`.
    -- Let's use `self.dropout2` explicitly and remove it from `ffn` sequential to be clearer.

    local out = self.res1 + ffn_out
    return out
end

function TransformerBlock:backward(gradOutput)
    local d_ffn_out = gradOutput
    local d_res1 = gradOutput

    -- Backward FFN (requires flatten)
    local B = gradOutput.shape[1]
    local S = gradOutput.shape[2]
    local D = gradOutput.shape[3]
    local d_ffn_out_flat = d_ffn_out:view({B * S, D})

    local d_h2_flat = self.ffn:backward(d_ffn_out_flat)
    local d_h2 = d_h2_flat:view({B, S, D})

    local d_res1_ln2 = self.ln2:backward(d_h2)

    -- d_res1 + d_res1_ln2
    local d_res1_sum = d_res1 + d_res1_ln2

    local d_attn_out = d_res1_sum
    local d_x = d_res1_sum

    d_attn_out = self.dropout1:backward(d_attn_out)

    local d_h = self.attn:backward(d_attn_out)
    local d_x_ln1 = self.ln1:backward(d_h)

    local d_x_sum = d_x + d_x_ln1
    return d_x_sum
end


local MiniLLM = setmetatable({}, {__index = Module})
MiniLLM.__index = MiniLLM

function MiniLLM:new(vocab_size, embed_dim, num_heads, num_layers, dropout_p)
    local self = setmetatable({}, MiniLLM)
    self:init()

    self.embedding = layers.Embedding:new(vocab_size, embed_dim)
    self.pos_embedding = layers.Embedding:new(128, embed_dim)
    self.dropout = layers.Dropout:new(dropout_p)

    self.blocks = layers.Sequential:new()
    for i = 1, num_layers do
        self.blocks:add_module(tostring(i), TransformerBlock:new(embed_dim, num_heads, dropout_p))
    end

    self.ln_f = layers.LayerNorm:new(embed_dim)
    self.head = layers.Linear:new(embed_dim, vocab_size)

    self:add_module("embedding", self.embedding)
    self:add_module("pos_embedding", self.pos_embedding)
    self:add_module("dropout", self.dropout)
    self:add_module("blocks", self.blocks)
    self:add_module("ln_f", self.ln_f)
    self:add_module("head", self.head)

    return self
end

function MiniLLM:forward(input)
    local batch = input.shape[1]
    local seq = input.shape[2]

    local tok_emb = self.embedding:forward(input)

    local pos_indices = Tensor.zeros({1, seq})
    for i=1, seq do pos_indices.data[i] = i end

    local pos_emb = self.pos_embedding:forward(pos_indices)

    -- Manually broadcast pos_emb (1, S, D) to (B, S, D)
    local x = Tensor.zeros(tok_emb.shape)
    local te_data = tok_emb.data
    local pe_data = pos_emb.data
    local x_data = x.data
    local dim = self.embedding.embedding_dim

    for b = 0, batch - 1 do
        for s = 0, seq - 1 do
            for d = 0, dim - 1 do
                local idx = (b * seq + s) * dim + d + 1
                local p_idx = s * dim + d + 1
                x_data[idx] = te_data[idx] + pe_data[p_idx]
            end
        end
    end

    x = self.dropout:forward(x)

    local mask = Tensor.zeros({seq, seq})
    for i=1, seq do
        for j=i+1, seq do
            mask:set({i, j}, -1e9)
        end
    end

    for _, name in ipairs(self.blocks._ordered_modules) do
        x = self.blocks._modules[name]:forward(x, mask)
    end

    x = self.ln_f:forward(x)

    -- Reshape x for head: (B*S, D)
    local x_flat = x:view({batch * seq, x.shape[3]})
    local logits = self.head:forward(x_flat)

    -- Reshape logits back to (B, S, V)
    local V = self.head.out_features
    return logits:view({batch, seq, V})
end

function MiniLLM:backward(gradOutput)
    local batch = gradOutput.shape[1]
    local seq = gradOutput.shape[2]
    local V = gradOutput.shape[3]

    local grad_flat = gradOutput:view({batch * seq, V})

    local d_x_flat = self.head:backward(grad_flat)
    local dim = self.embedding.embedding_dim
    local d_x = d_x_flat:view({batch, seq, dim})

    d_x = self.ln_f:backward(d_x)

    for i = #self.blocks._ordered_modules, 1, -1 do
        local name = self.blocks._ordered_modules[i]
        d_x = self.blocks._modules[name]:backward(d_x)
    end

    d_x = self.dropout:backward(d_x)

    local d_tok = d_x

    local d_pos_t = Tensor.zeros({1, seq, dim})
    local dx_data = d_x.data
    local dp_data = d_pos_t.data

    for b = 0, batch - 1 do
        for s = 0, seq - 1 do
            for d = 0, dim - 1 do
                local idx = (b * seq + s) * dim + d + 1
                local p_idx = s * dim + d + 1
                dp_data[p_idx] = dp_data[p_idx] + dx_data[idx]
            end
        end
    end

    self.embedding:backward(d_tok)
    self.pos_embedding:backward(d_pos_t)

    return nil
end

return {
    TransformerBlock = TransformerBlock,
    MiniLLM = MiniLLM
}

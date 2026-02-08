local Module = require("luanet.module")
local Tensor = require("luanet.tensor")
local Linear = require("luanet.layers.linear")
local activations = require("luanet.layers.activations")

local MultiHeadAttention = setmetatable({}, {__index = Module})
MultiHeadAttention.__index = MultiHeadAttention

function MultiHeadAttention:new(embed_dim, num_heads)
    local self = setmetatable({}, MultiHeadAttention)
    self:init()

    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim / num_heads

    if self.head_dim * num_heads ~= embed_dim then
        error("embed_dim must be divisible by num_heads")
    end

    self.q_proj = Linear:new(embed_dim, embed_dim)
    self.k_proj = Linear:new(embed_dim, embed_dim)
    self.v_proj = Linear:new(embed_dim, embed_dim)
    self.out_proj = Linear:new(embed_dim, embed_dim)

    self:add_module("q_proj", self.q_proj)
    self:add_module("k_proj", self.k_proj)
    self:add_module("v_proj", self.v_proj)
    self:add_module("out_proj", self.out_proj)

    self.softmax_layer = activations.Softmax:new(2)
    -- We don't register softmax_layer as module because we use it manually

    return self
end

function MultiHeadAttention:forward(query, key, value, mask)
    local batch_size = query.shape[1]
    local seq_len_q = query.shape[2]
    local seq_len_k = key.shape[2]
    local dim = self.embed_dim

    self.input_q = query
    self.input_k = key
    self.input_v = value

    local q_flat = query:view({batch_size * seq_len_q, dim})
    local k_flat = key:view({batch_size * seq_len_k, dim})
    local v_flat = value:view({batch_size * seq_len_k, dim})

    local q = self.q_proj:forward(q_flat)
    local k = self.k_proj:forward(k_flat)
    local v = self.v_proj:forward(v_flat)

    local num_heads = self.num_heads
    local head_dim = self.head_dim
    local scale = 1.0 / math.sqrt(head_dim)

    local output_flat = Tensor.zeros({batch_size * seq_len_q, dim})
    local o_data = output_flat.data
    local q_data = q.data
    local k_data = k.data
    local v_data = v.data

    self.cache = {}

    for b = 0, batch_size - 1 do
        for h = 0, num_heads - 1 do
            local cache_idx = b * num_heads + h + 1
            local cache_item = {}

            local q_head = Tensor.zeros({seq_len_q, head_dim})
            local k_head = Tensor.zeros({seq_len_k, head_dim})
            local v_head = Tensor.zeros({seq_len_k, head_dim})

            -- Extract heads
            for s = 0, seq_len_q - 1 do
                for d = 0, head_dim - 1 do
                    local flat_idx = (b * seq_len_q + s) * dim + (h * head_dim + d) + 1
                    q_head:set({s+1, d+1}, q_data[flat_idx])
                end
            end
            for s = 0, seq_len_k - 1 do
                for d = 0, head_dim - 1 do
                    local flat_idx = (b * seq_len_k + s) * dim + (h * head_dim + d) + 1
                    k_head:set({s+1, d+1}, k_data[flat_idx])
                    v_head:set({s+1, d+1}, v_data[flat_idx])
                end
            end

            cache_item.q_head = q_head
            cache_item.k_head = k_head
            cache_item.v_head = v_head

            local scores = q_head:matmul(k_head:transpose())
            scores = scores:mul(scale)

            if mask then
                -- Assume mask is causal (seq_q, seq_k) and broadcasted
                scores = scores + mask
            end

            local attn_probs = self.softmax_layer:forward(scores)
            cache_item.attn_probs = attn_probs

            local context = attn_probs:matmul(v_head)

            for s = 0, seq_len_q - 1 do
                for d = 0, head_dim - 1 do
                    local flat_idx = (b * seq_len_q + s) * dim + (h * head_dim + d) + 1
                    o_data[flat_idx] = context:get({s+1, d+1})
                end
            end

            self.cache[cache_idx] = cache_item
        end
    end

    local output = self.out_proj:forward(output_flat)
    return output:view({batch_size, seq_len_q, dim})
end

function MultiHeadAttention:backward(gradOutput)
    local batch_size = self.input_q.shape[1]
    local seq_len_q = self.input_q.shape[2]
    local seq_len_k = self.input_k.shape[2]
    local dim = self.embed_dim
    local num_heads = self.num_heads
    local head_dim = self.head_dim
    local scale = 1.0 / math.sqrt(head_dim)

    local grad_out_flat = gradOutput:view({batch_size * seq_len_q, dim})
    local grad_concat = self.out_proj:backward(grad_out_flat) -- (B*S, D)
    local gc_data = grad_concat.data

    local dq_flat = Tensor.zeros({batch_size * seq_len_q, dim})
    local dk_flat = Tensor.zeros({batch_size * seq_len_k, dim})
    local dv_flat = Tensor.zeros({batch_size * seq_len_k, dim})

    local dq_data = dq_flat.data
    local dk_data = dk_flat.data
    local dv_data = dv_flat.data

    for b = 0, batch_size - 1 do
        for h = 0, num_heads - 1 do
            local cache_idx = b * num_heads + h + 1
            local item = self.cache[cache_idx]
            local q_head = item.q_head
            local k_head = item.k_head
            local v_head = item.v_head
            local probs = item.attn_probs

            -- Extract grad_context for this head
            local grad_context = Tensor.zeros({seq_len_q, head_dim})
            for s = 0, seq_len_q - 1 do
                for d = 0, head_dim - 1 do
                    local flat_idx = (b * seq_len_q + s) * dim + (h * head_dim + d) + 1
                    grad_context:set({s+1, d+1}, gc_data[flat_idx])
                end
            end

            -- d_probs = grad_context * v_head^T
            local d_probs = grad_context:matmul(v_head:transpose())

            -- d_v_head = probs^T * grad_context
            local d_v_head = probs:transpose():matmul(grad_context)

            -- d_scores = softmax_backward(d_probs, probs)
            -- dSi = Si * (dLi - sum(dLk * Sk))
            -- d_probs is dLi.
            -- dot = sum(d_probs * probs, dim=2)
            local dot = d_probs:mul(probs):sum(2) -- (seq_q, 1)
            local d_scores = probs:mul(d_probs - dot) -- broadcasting dot

            d_scores = d_scores:mul(scale)

            -- d_q_head = d_scores * k_head
            local d_q_head = d_scores:matmul(k_head)

            -- d_k_head = d_scores^T * q_head
            local d_k_head = d_scores:transpose():matmul(q_head)

            -- Accumulate into flat tensors
            for s = 0, seq_len_q - 1 do
                for d = 0, head_dim - 1 do
                    local flat_idx = (b * seq_len_q + s) * dim + (h * head_dim + d) + 1
                    dq_data[flat_idx] = dq_data[flat_idx] + d_q_head:get({s+1, d+1})
                end
            end
            for s = 0, seq_len_k - 1 do
                for d = 0, head_dim - 1 do
                    local flat_idx = (b * seq_len_k + s) * dim + (h * head_dim + d) + 1
                    dk_data[flat_idx] = dk_data[flat_idx] + d_k_head:get({s+1, d+1})
                    dv_data[flat_idx] = dv_data[flat_idx] + d_v_head:get({s+1, d+1})
                end
            end
        end
    end

    local d_q_in = self.q_proj:backward(dq_flat)
    local d_k_in = self.k_proj:backward(dk_flat)
    local d_v_in = self.v_proj:backward(dv_flat)

    if self.input_q == self.input_k and self.input_q == self.input_v then
        -- Self attention
        local grad_sum = d_q_in + d_k_in + d_v_in
        return grad_sum:view({batch_size, seq_len_q, dim})
    else
        -- Return table of grads? Or error?
        -- Module backward typically returns grad w.r.t input.
        -- But here we have 3 inputs.
        -- Usually in MHA(query, key, value), we return (dq, dk, dv).
        -- But standard Module:backward returns 1 tensor.
        -- If wrapped in a larger model, we need to handle this.
        -- For TransformerBlock, Q=K=V=input. So return sum.
        return d_q_in + d_k_in + d_v_in -- Shapes match B*S*D? Yes.
        -- d_q_in is (B*S, D). Reshape?
    end
end

return MultiHeadAttention

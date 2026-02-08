local Module = require("luanet.module")
local Tensor = require("luanet.tensor")

local Embedding = setmetatable({}, {__index = Module})
Embedding.__index = Embedding

function Embedding:new(num_embeddings, embedding_dim)
    local self = setmetatable({}, Embedding)
    self:init()

    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim

    self.weight = Tensor.randn({num_embeddings, embedding_dim})
    self.gradWeight = Tensor.zeros({num_embeddings, embedding_dim})

    self:register_parameter("weight", self.weight, self.gradWeight)

    return self
end

function Embedding:forward(input)
    self.input = input
    -- input shape: (batch, seq_len) or (batch)

    local batch = input.shape[1]
    local seq_len = 1
    if #input.shape > 1 then seq_len = input.shape[2] end

    local out_shape
    if #input.shape == 1 then
        out_shape = {batch, self.embedding_dim}
    else
        out_shape = {batch, seq_len, self.embedding_dim}
    end

    local output = Tensor.zeros(out_shape)

    local flat_input = input.data
    local out_data = output.data
    local dim = self.embedding_dim
    local weight_data = self.weight.data

    for i = 1, #flat_input do
        local idx = flat_input[i]
        if idx < 1 or idx > self.num_embeddings then
            error("Embedding index out of bounds: " .. idx)
        end

        local w_start = (idx - 1) * dim + 1
        local o_start = (i - 1) * dim + 1

        for k = 0, dim - 1 do
            out_data[o_start + k] = weight_data[w_start + k]
        end
    end

    return output
end

function Embedding:backward(gradOutput)
    local flat_input = self.input.data
    local grad_data = gradOutput.data
    local gw_data = self.gradWeight.data
    local dim = self.embedding_dim

    for i = 1, #flat_input do
        local idx = flat_input[i]

        local w_start = (idx - 1) * dim + 1
        local g_start = (i - 1) * dim + 1

        for k = 0, dim - 1 do
            gw_data[w_start + k] = gw_data[w_start + k] + grad_data[g_start + k]
        end
    end

    return nil
end

return Embedding

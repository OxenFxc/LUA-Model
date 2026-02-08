local Module = require("luanet.module")
local Tensor = require("luanet.tensor")

local LayerNorm = setmetatable({}, {__index = Module})
LayerNorm.__index = LayerNorm

function LayerNorm:new(normalized_shape, eps)
    local self = setmetatable({}, LayerNorm)
    self:init()

    self.normalized_shape = normalized_shape
    self.eps = eps or 1e-5

    -- weight (gamma) initialized to 1
    self.weight = Tensor.new(nil, {normalized_shape})
    local w_data = {}
    for i=1, normalized_shape do w_data[i] = 1 end
    self.weight.data = w_data
    self.weight.strides = {1}

    self.gradWeight = Tensor.zeros({normalized_shape})

    -- bias (beta) initialized to 0
    self.bias = Tensor.zeros({normalized_shape})
    self.gradBias = Tensor.zeros({normalized_shape})

    self:register_parameter("weight", self.weight, self.gradWeight)
    self:register_parameter("bias", self.bias, self.gradBias)

    return self
end

function LayerNorm:forward(input)
    self.input = input
    local dim = self.normalized_shape
    if input.shape[#input.shape] ~= dim then
        error("LayerNorm shape mismatch: input last dim " .. input.shape[#input.shape] .. " vs " .. dim)
    end

    local output = Tensor.zeros(input.shape)

    local flat_in = input.data
    local flat_out = output.data
    local w = self.weight.data
    local b = self.bias.data
    local eps = self.eps

    self.means = {}
    self.inv_stds = {}

    local num_samples = math.floor(#flat_in / dim)

    for i = 0, num_samples - 1 do
        local start = i * dim + 1

        -- Compute mean
        local sum = 0
        for k = 0, dim - 1 do
            sum = sum + flat_in[start + k]
        end
        local mean = sum / dim
        self.means[i+1] = mean

        -- Compute var
        local sum_sq = 0
        for k = 0, dim - 1 do
            local d = flat_in[start + k] - mean
            sum_sq = sum_sq + d * d
        end
        local var = sum_sq / dim
        local inv_std = 1 / math.sqrt(var + eps)
        self.inv_stds[i+1] = inv_std

        -- Normalize and scale
        for k = 0, dim - 1 do
            local val = (flat_in[start + k] - mean) * inv_std
            flat_out[start + k] = val * w[k+1] + b[k+1]
        end
    end

    return output
end

function LayerNorm:backward(gradOutput)
    local dim = self.normalized_shape
    local flat_grad = gradOutput.data
    local flat_in = self.input.data
    local w = self.weight.data

    local gradInput = Tensor.zeros(self.input.shape)
    local flat_gin = gradInput.data

    local gw = self.gradWeight.data
    local gb = self.gradBias.data

    local num_samples = math.floor(#flat_grad / dim)

    for i = 0, num_samples - 1 do
        local start = i * dim + 1
        local mean = self.means[i+1]
        local inv_std = self.inv_stds[i+1]

        local d_x_hat = {}
        local sum_dxhat = 0
        local sum_dxhat_xhat = 0

        for k = 0, dim - 1 do
            local dy = flat_grad[start + k]
            local x_hat = (flat_in[start + k] - mean) * inv_std

            -- Accumulate gradients
            gw[k+1] = gw[k+1] + dy * x_hat
            gb[k+1] = gb[k+1] + dy

            local dxh = dy * w[k+1]
            d_x_hat[k+1] = dxh

            sum_dxhat = sum_dxhat + dxh
            sum_dxhat_xhat = sum_dxhat_xhat + dxh * x_hat
        end

        for k = 0, dim - 1 do
            local x_hat = (flat_in[start + k] - mean) * inv_std
            local dxh = d_x_hat[k+1]
            local dx = inv_std * (dxh - sum_dxhat / dim - x_hat * sum_dxhat_xhat / dim)
            flat_gin[start + k] = dx
        end
    end

    return gradInput
end

return LayerNorm

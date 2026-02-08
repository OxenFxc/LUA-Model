local Tensor = require("luanet.tensor")

local Loss = {}

-- MSELoss --
local MSELoss = {}
MSELoss.__index = MSELoss

function MSELoss:new()
    local self = setmetatable({}, MSELoss)
    return self
end

function MSELoss:forward(input, target)
    self.input = input
    self.target = target
    local diff = input - target
    local sq = diff:apply(function(x) return x*x end)
    -- mean reduction
    return sq:sum() / #sq.data
end

function MSELoss:backward()
    local diff = self.input - self.target
    local n = #diff.data
    -- gradient of mean(sq) = 2 * (x - y) / n
    -- Wait, Tensor:mul returns new tensor.
    -- We want to avoid creating too many tensors if possible but here it's fine.
    -- diff:mul(2/n)
    return diff:mul(2 / n)
end

Loss.MSELoss = MSELoss

-- CrossEntropyLoss --
local CrossEntropyLoss = {}
CrossEntropyLoss.__index = CrossEntropyLoss

function CrossEntropyLoss:new()
    local self = setmetatable({}, CrossEntropyLoss)
    return self
end

function CrossEntropyLoss:forward(input, target)
    -- Input: (N, C) logits
    -- Target: (N) indices (1-based)
    self.input = input
    self.target = target
    local N = input.shape[1]
    local C = input.shape[2]

    local loss = 0
    self.probs = Tensor.zeros(input.shape)

    for i = 1, N do
        local row = {}
        local max_val = -math.huge
        -- Find max for numerical stability
        for j = 1, C do
            local val = input:get({i, j})
            if val > max_val then max_val = val end
        end

        local sum_exp = 0
        for j = 1, C do
            local val = input:get({i, j})
            row[j] = math.exp(val - max_val)
            sum_exp = sum_exp + row[j]
        end

        for j = 1, C do
            self.probs:set({i, j}, row[j] / sum_exp)
        end

        -- Target access: target should be Tensor (N) or (N, 1) or Lua table?
        -- Assume Tensor (N)
        local target_idx
        if #self.target.shape == 1 then
            target_idx = self.target:get({i})
        else
            target_idx = self.target:get({i, 1})
        end

        -- Check bounds
        if target_idx < 1 or target_idx > C then
            error("Target index out of bounds: " .. target_idx)
        end

        local val = input:get({i, target_idx})
        loss = loss - val + max_val + math.log(sum_exp)
    end

    return loss / N
end

function CrossEntropyLoss:backward()
    local N = self.input.shape[1]
    local grad = self.probs:clone()

    for i = 1, N do
        local target_idx
        if #self.target.shape == 1 then
            target_idx = self.target:get({i})
        else
            target_idx = self.target:get({i, 1})
        end

        local val = grad:get({i, target_idx})
        grad:set({i, target_idx}, val - 1)
    end

    return grad:div(N)
end

Loss.CrossEntropyLoss = CrossEntropyLoss

return Loss

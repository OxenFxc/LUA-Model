local Optim = {}

Optim.Adam = require("luanet.optim.adam")

local Tensor = require("luanet.tensor")

local SGD = {}
SGD.__index = SGD

function SGD:new(params, grads, lr, momentum)
    local self = setmetatable({}, SGD)
    self.params = params
    self.grads = grads
    self.lr = lr or 0.01
    self.momentum = momentum or 0
    self.velocities = {}

    return self
end

function SGD:step()
    for i = 1, #self.params do
        local param = self.params[i]
        local grad = self.grads[i]

        if grad then
            if self.momentum ~= 0 then
                if not self.velocities[i] then
                    self.velocities[i] = Tensor.zeros(param.shape)
                end
                local v = self.velocities[i]

                for k = 1, #param.data do
                    v.data[k] = self.momentum * v.data[k] + grad.data[k]
                    param.data[k] = param.data[k] - self.lr * v.data[k]
                end
            else
                for k = 1, #param.data do
                    param.data[k] = param.data[k] - self.lr * grad.data[k]
                end
            end
        end
    end
end

function SGD:zero_grad()
    for _, g in ipairs(self.grads) do
        if g then g:zero() end
    end
end

Optim.SGD = SGD

return Optim

local Tensor = require("luanet.tensor")

local Adam = {}
Adam.__index = Adam

function Adam:new(params, grads, lr, betas, eps, weight_decay)
    local self = setmetatable({}, Adam)
    self.params = params
    self.grads = grads
    self.lr = lr or 0.001
    self.betas = betas or {0.9, 0.999}
    self.eps = eps or 1e-8
    self.weight_decay = weight_decay or 0

    self.m = {}
    self.v = {}
    self.t = 0

    -- Initialize moments
    for i = 1, #self.params do
        self.m[i] = Tensor.zeros(self.params[i].shape)
        self.v[i] = Tensor.zeros(self.params[i].shape)
    end

    return self
end

function Adam:step()
    self.t = self.t + 1
    local beta1, beta2 = self.betas[1], self.betas[2]
    local lr = self.lr
    local eps = self.eps

    local bias_correction1 = 1 - beta1^self.t
    local bias_correction2 = 1 - beta2^self.t

    for i = 1, #self.params do
        local param = self.params[i]
        local grad = self.grads[i]

        if grad then
            local m = self.m[i]
            local v = self.v[i]

            local size = #param.data
            local p_data = param.data
            local g_data = grad.data
            local m_data = m.data
            local v_data = v.data
            local wd = self.weight_decay

            for k = 1, size do
                local g = g_data[k]
                if wd > 0 then
                    g = g + wd * p_data[k]
                end

                -- Update moments
                m_data[k] = beta1 * m_data[k] + (1 - beta1) * g
                v_data[k] = beta2 * v_data[k] + (1 - beta2) * g * g

                local m_hat = m_data[k] / bias_correction1
                local v_hat = v_data[k] / bias_correction2
                local denom = math.sqrt(v_hat) + eps

                p_data[k] = p_data[k] - lr * m_hat / denom
            end
        end
    end
end

function Adam:zero_grad()
    for _, g in ipairs(self.grads) do
        if g then g:zero() end
    end
end

return Adam

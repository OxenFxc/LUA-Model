local Tensor = require("luanet.tensor")

local GradClip = {}

function GradClip.clip_grad_norm(gradients, max_norm, norm_type)
    -- gradients: list of gradient tensors
    max_norm = max_norm or 1.0
    norm_type = norm_type or 2.0

    local total_norm = 0
    if norm_type == 2.0 then
        for _, g in ipairs(gradients) do
            if g then
                -- sum(g^2)
                -- Avoid full pow(2) tensor if possible for speed?
                -- sum(g*g)
                local d = g.data
                local s = 0
                for i=1, #d do s = s + d[i]*d[i] end
                total_norm = total_norm + s
            end
        end
        total_norm = math.sqrt(total_norm)
    else
        error("Only L2 norm supported for now")
    end

    local clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1 then
        for _, g in ipairs(gradients) do
            if g then
                local d = g.data
                for i = 1, #d do
                    d[i] = d[i] * clip_coef
                end
            end
        end
    end

    return total_norm
end

return GradClip
